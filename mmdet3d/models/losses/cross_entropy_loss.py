# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    """Lightweight CrossEntropyLoss compatible with MMDetection config fields.

    This CPU-oriented fallback avoids importing ``mmdet`` just to build the
    loss module for inference-capable models such as PointNet2.
    """

    def __init__(self,
                 use_sigmoid: bool = False,
                 class_weight: Optional[Sequence[float]] = None,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 avg_non_ignore: bool = False) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        if class_weight is None:
            self.class_weight = None
        else:
            self.register_buffer(
                'class_weight',
                torch.tensor(class_weight, dtype=torch.float32),
                persistent=False)

    def forward(self,
                pred: Tensor,
                target: Tensor,
                ignore_index: Optional[int] = None,
                **kwargs) -> Tensor:
        del kwargs
        if self.use_sigmoid:
            if target.shape != pred.shape:
                target = F.one_hot(target.long(), num_classes=pred.shape[1])
                target = target.permute(0, 2, 1).to(pred.dtype)
            loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction='none')
            if self.reduction == 'sum':
                loss = loss.sum()
            elif self.reduction == 'mean':
                loss = loss.mean()
        else:
            weight = getattr(self, 'class_weight', None)
            loss = F.cross_entropy(
                pred,
                target.long(),
                weight=weight,
                reduction=self.reduction,
                ignore_index=-100 if ignore_index is None else ignore_index)
            if self.avg_non_ignore and self.reduction == 'mean' \
                    and ignore_index is not None:
                valid = (target != ignore_index).sum().clamp_min(1)
                loss = loss * target.numel() / valid

        return loss * self.loss_weight
