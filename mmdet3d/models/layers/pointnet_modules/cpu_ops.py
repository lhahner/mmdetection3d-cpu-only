# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch import nn


def gather_points(features: Tensor, indices: Tensor) -> Tensor:
    """Gather points from ``features`` using batched indices.

    Args:
        features (Tensor): Shape (B, C, N).
        indices (Tensor): Shape (B, M).

    Returns:
        Tensor: Gathered features with shape (B, C, M).
    """
    index = indices.long().unsqueeze(1).expand(-1, features.shape[1], -1)
    return torch.gather(features, 2, index)


def grouping_operation(features: Tensor, indices: Tensor) -> Tensor:
    """Group point features using batched indices.

    Args:
        features (Tensor): Shape (B, C, N).
        indices (Tensor): Shape (B, M, K).

    Returns:
        Tensor: Grouped features with shape (B, C, M, K).
    """
    expanded = indices.long().unsqueeze(1).expand(-1, features.shape[1], -1,
                                                  -1)
    features = features.unsqueeze(2).expand(-1, -1, indices.shape[1], -1)
    return torch.gather(features, 3, expanded)


def three_nn(target: Tensor, source: Tensor) -> Tuple[Tensor, Tensor]:
    """Find the three nearest source points for each target point."""
    dists = torch.cdist(target, source)
    dists, idx = torch.topk(dists, k=3, dim=-1, largest=False, sorted=False)
    return dists, idx.long()


def three_interpolate(source_feats: Tensor, indices: Tensor,
                      weight: Tensor) -> Tensor:
    """Interpolate source features to target points using weighted neighbors."""
    grouped = grouping_operation(source_feats, indices)
    return torch.sum(grouped * weight.unsqueeze(1), dim=-1)


def _square_distance(src: Tensor, dst: Tensor) -> Tensor:
    return torch.cdist(src, dst)**2


def _furthest_point_sample(xyz: Tensor, num_points: int) -> Tensor:
    """Pure PyTorch furthest point sampling for D-FPS."""
    device = xyz.device
    batch_size, num_total, _ = xyz.shape
    centroids = torch.zeros(batch_size, num_points, dtype=torch.long,
                            device=device)
    distance = torch.full((batch_size, num_total), float('inf'),
                          device=device)
    farthest = torch.zeros(batch_size, dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, device=device)

    for i in range(num_points):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid)**2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=1).indices

    return centroids


class PointsSampler(nn.Module):
    """Minimal CPU fallback for PointNet2 FPS-based sampling.

    Supports the sampling modes used by this repository's PointNet2 configs.
    """

    def __init__(self, num_point: Sequence[int], fps_mod_list: Sequence[str],
                 fps_sample_range_list: Sequence[int]) -> None:
        super().__init__()
        self.num_point = list(num_point)
        self.fps_mod_list = list(fps_mod_list)
        self.fps_sample_range_list = list(fps_sample_range_list)

    def forward(self, points_xyz: Tensor,
                features: Optional[Tensor] = None) -> Tensor:
        sampled_parts: List[Tensor] = []
        last_end = 0
        batch_size = points_xyz.shape[0]

        for num_point, fps_mod, sample_range in zip(self.num_point,
                                                    self.fps_mod_list,
                                                    self.fps_sample_range_list):
            if fps_mod != 'D-FPS':
                raise NotImplementedError(
                    f'CPU fallback only supports D-FPS, got {fps_mod!r}.')

            end = points_xyz.shape[1] if sample_range < 0 else sample_range
            xyz_slice = points_xyz[:, last_end:end, :]
            if xyz_slice.numel() == 0:
                raise ValueError('PointsSampler received an empty point slice.')

            cur_idx = _furthest_point_sample(xyz_slice.contiguous(), num_point)
            sampled_parts.append(cur_idx + last_end)
            last_end = end

        return torch.cat(sampled_parts, dim=1).reshape(batch_size, -1)


class GroupAll(nn.Module):

    def __init__(self, use_xyz: bool = True) -> None:
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self,
                points_xyz: Tensor,
                new_xyz: Optional[Tensor],
                features: Optional[Tensor] = None) -> Tensor:
        grouped_xyz = points_xyz.transpose(1, 2).unsqueeze(2)
        if features is None:
            if not self.use_xyz:
                raise AssertionError('Cannot disable xyz when features are None')
            return grouped_xyz
        grouped_features = features.unsqueeze(2)
        if self.use_xyz:
            return torch.cat([grouped_xyz, grouped_features], dim=1)
        return grouped_features


class QueryAndGroup(nn.Module):

    def __init__(self,
                 radius: float,
                 sample_num: int,
                 min_radius: float = 0.0,
                 use_xyz: bool = True,
                 normalize_xyz: bool = False,
                 return_grouped_xyz: bool = False,
                 return_grouped_idx: bool = False) -> None:
        super().__init__()
        self.radius = radius
        self.sample_num = sample_num
        self.min_radius = min_radius
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz
        self.return_grouped_xyz = return_grouped_xyz
        self.return_grouped_idx = return_grouped_idx

    def forward(self,
                points_xyz: Tensor,
                new_xyz: Tensor,
                features: Optional[Tensor] = None):
        sqrdists = _square_distance(new_xyz, points_xyz)
        mask = (sqrdists <= self.radius**2)
        if self.min_radius > 0:
            mask = mask & (sqrdists >= self.min_radius**2)

        batch_size, num_query, num_points = mask.shape
        grouped_idx = torch.zeros((batch_size, num_query, self.sample_num),
                                  dtype=torch.long,
                                  device=points_xyz.device)

        for b in range(batch_size):
            for m in range(num_query):
                valid = torch.nonzero(mask[b, m], as_tuple=False).flatten()
                if valid.numel() == 0:
                    valid = torch.zeros(1, dtype=torch.long,
                                        device=points_xyz.device)
                if valid.numel() >= self.sample_num:
                    chosen = valid[:self.sample_num]
                else:
                    pad = valid[-1].repeat(self.sample_num - valid.numel())
                    chosen = torch.cat([valid, pad], dim=0)
                grouped_idx[b, m] = chosen

        grouped_xyz = grouping_operation(points_xyz.transpose(1, 2),
                                         grouped_idx)
        grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz = grouped_xyz / self.radius

        results = []
        if features is not None:
            grouped_features = grouping_operation(features, grouped_idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            if not self.use_xyz:
                raise AssertionError('Cannot have not features and not use xyz')
            new_features = grouped_xyz

        results.append(new_features)
        if self.return_grouped_xyz:
            results.append(grouped_xyz)
        if self.return_grouped_idx:
            results.append(grouped_idx)

        if len(results) == 1:
            return results[0]
        return tuple(results)


def ball_query(min_radius: float, radius: float, sample_num: int, xyz: Tensor,
               new_xyz: Tensor, xyz_batch_cnt: Tensor,
               new_xyz_batch_cnt: Tensor) -> Tensor:
    """Stacked ball query CPU fallback."""
    indices: List[Tensor] = []
    xyz_start = 0
    new_xyz_start = 0
    for xyz_count, new_xyz_count in zip(xyz_batch_cnt.tolist(),
                                        new_xyz_batch_cnt.tolist()):
        xyz_slice = xyz[xyz_start:xyz_start + xyz_count]
        new_xyz_slice = new_xyz[new_xyz_start:new_xyz_start + new_xyz_count]
        sqrdists = torch.cdist(new_xyz_slice, xyz_slice)**2
        mask = sqrdists <= radius**2
        if min_radius > 0:
            mask = mask & (sqrdists >= min_radius**2)

        batch_idx = torch.empty((new_xyz_count, sample_num),
                                dtype=torch.long,
                                device=xyz.device)
        for i in range(new_xyz_count):
            valid = torch.nonzero(mask[i], as_tuple=False).flatten()
            if valid.numel() == 0:
                batch_idx[i] = -1
                continue
            if valid.numel() >= sample_num:
                chosen = valid[:sample_num]
            else:
                pad = valid[-1].repeat(sample_num - valid.numel())
                chosen = torch.cat([valid, pad], dim=0)
            batch_idx[i] = chosen + xyz_start
        indices.append(batch_idx)
        xyz_start += xyz_count
        new_xyz_start += new_xyz_count
    return torch.cat(indices, dim=0)


def grouping_operation_stacked(features: Tensor, indices: Tensor,
                               xyz_batch_cnt: Tensor,
                               new_xyz_batch_cnt: Tensor) -> Tensor:
    """Stacked grouping op CPU fallback."""
    del xyz_batch_cnt, new_xyz_batch_cnt
    safe_idx = indices.clamp_min(0)
    gathered = features[safe_idx.long()]
    return gathered.permute(0, 2, 1).contiguous()
