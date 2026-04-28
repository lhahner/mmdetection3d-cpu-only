# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import abstractmethod
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmdet3d.structures.points import BasePoints
from .utils import limit_period

try:
    from mmcv.ops import box_iou_rotated, points_in_boxes_all, points_in_boxes_part
    MMCV_OPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    box_iou_rotated = None
    points_in_boxes_all = None
    points_in_boxes_part = None
    MMCV_OPS_AVAILABLE = False


def _rotated_box_to_corners(box: Tensor) -> Tensor:
    """Convert a single XYWHR box to 4 BEV corners."""
    cx, cy, width, height, yaw = box
    half_w = width / 2
    half_h = height / 2
    corners = box.new_tensor([[-half_w, -half_h], [-half_w, half_h],
                              [half_w, half_h], [half_w, -half_h]])
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)
    rot = box.new_tensor([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
    return corners @ rot + box.new_tensor([cx, cy])


def _polygon_area(points: List[Tensor]) -> Tensor:
    """Compute polygon area with the shoelace formula."""
    if len(points) < 3:
        return torch.tensor(0.0, dtype=points[0].dtype,
                            device=points[0].device) if points else torch.tensor(0.0)
    area = points[0].new_tensor(0.0)
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        area = area + point[0] * next_point[1] - next_point[0] * point[1]
    return area.abs() * 0.5


def _cross_2d(vec1: Tensor, vec2: Tensor) -> Tensor:
    return vec1[0] * vec2[1] - vec1[1] * vec2[0]


def _is_inside(point: Tensor, edge_start: Tensor, edge_end: Tensor) -> bool:
    return _cross_2d(edge_end - edge_start, point - edge_start) >= 0


def _line_intersection(p1: Tensor, p2: Tensor, q1: Tensor, q2: Tensor) -> Tensor:
    r = p2 - p1
    s = q2 - q1
    denom = _cross_2d(r, s)
    if torch.abs(denom) < 1e-8:
        return (p2 + q1) / 2
    t = _cross_2d(q1 - p1, s) / denom
    return p1 + t * r


def _intersect_polygons(subject: Tensor, clip: Tensor) -> List[Tensor]:
    """Clip one convex polygon with another."""
    output = [point for point in subject]
    for idx in range(len(clip)):
        edge_start = clip[idx]
        edge_end = clip[(idx + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        previous = input_list[-1]
        for current in input_list:
            current_inside = _is_inside(current, edge_start, edge_end)
            previous_inside = _is_inside(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, edge_start,
                                           edge_end))
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def _box_iou_rotated_cpu(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    ious = boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    corners1 = [_rotated_box_to_corners(box) for box in boxes1]
    corners2 = [_rotated_box_to_corners(box) for box in boxes2]
    areas1 = boxes1[:, 2] * boxes1[:, 3]
    areas2 = boxes2[:, 2] * boxes2[:, 3]

    for row, poly1 in enumerate(corners1):
        for col, poly2 in enumerate(corners2):
            intersection = _intersect_polygons(poly1, poly2)
            if not intersection:
                continue
            inter_area = _polygon_area(intersection)
            union = areas1[row] + areas2[col] - inter_area
            if union > 0:
                ious[row, col] = inter_area / union
    return ious


def _points_in_single_box(points: Tensor, corners: Tensor) -> Tensor:
    """Check whether points lie inside a single oriented box from corners."""
    origin = corners[0]
    axis1 = corners[1] - origin
    axis2 = corners[3] - origin
    axis3 = corners[4] - origin
    rel_points = points - origin

    proj1 = rel_points @ axis1
    proj2 = rel_points @ axis2
    proj3 = rel_points @ axis3

    axis1_norm = axis1 @ axis1
    axis2_norm = axis2 @ axis2
    axis3_norm = axis3 @ axis3

    return ((proj1 >= 0) & (proj1 <= axis1_norm) & (proj2 >= 0)
            & (proj2 <= axis2_norm) & (proj3 >= 0) & (proj3 <= axis3_norm))


class BaseInstance3DBoxes:
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in the
        box is (0.5, 0.5, 0).

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
            data with shape (N, box_dim).
        box_dim (int): Number of the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw). Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation. If False, the
            value of yaw will be set to 0 as minmax boxes. Defaults to True.
        origin (Tuple[float]): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    YAW_AXIS: int = 0

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float, float, float] = (0.5, 0.5, 0)
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, \
            ('The box dimension must be 2 and the length of the last '
             f'dimension must be {box_dim}, but got boxes with shape '
             f'{tensor.shape}.')

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding 0 as
            # a fake yaw and set with_yaw to False
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def shape(self) -> torch.Size:
        """torch.Size: Shape of boxes."""
        return self.tensor.shape

    @property
    def volume(self) -> Tensor:
        """Tensor: A vector with volume of each box in shape (N, )."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self) -> Tensor:
        """Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self) -> Tensor:
        """Tensor: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]

    @property
    def height(self) -> Tensor:
        """Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 5]

    @property
    def top_height(self) -> Tensor:
        """Tensor: A vector with top height of each box in shape (N, )."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self) -> Tensor:
        """Tensor: A vector with bottom height of each box in shape (N, )."""
        return self.tensor[:, 2]

    @property
    def center(self) -> Tensor:
        """Calculate the center of all the boxes.

        Note:
            In MMDetection3D's convention, the bottom center is usually taken
            as the default center.

            The relative position of the centers in different kinds of boxes
            are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar. It is
            recommended to use ``bottom_center`` or ``gravity_center`` for
            clearer usage.

        Returns:
            Tensor: A tensor with center of each box in shape (N, 3).
        """
        return self.bottom_center

    @property
    def bottom_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self) -> Tensor:
        """Tensor: A tensor with 8 corners of each box in shape (N, 8, 3)."""
        pass

    @property
    def bev(self) -> Tensor:
        """Tensor: 2D BEV box of each box with rotation in XYWHR format, in
        shape (N, 5)."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self) -> Tensor:
        """Tensor: A tensor of 2D BEV box of each box without rotation."""
        # Obtain BEV boxes with rotation in XYWHR format
        bev_rotated_boxes = self.bev
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

        # find the center of boxes
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(conditions, bev_rotated_boxes[:,
                                                                [0, 1, 3, 2]],
                                  bev_rotated_boxes[:, :4])

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    def in_range_bev(
            self, box_range: Union[Tensor, np.ndarray,
                                   Sequence[float]]) -> Tensor:
        """Check whether the boxes are in the given range.

        Args:
            box_range (Tensor or np.ndarray or Sequence[float]): The range of
                box in order of (x_min, y_min, x_max, y_max).

        Note:
            The original implementation of SECOND checks whether boxes in a
            range by checking whether the points are in a convex polygon, we
            reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each box is inside the
            reference range.
        """
        in_range_flags = ((self.bev[:, 0] > box_range[0])
                          & (self.bev[:, 1] > box_range[1])
                          & (self.bev[:, 0] < box_range[2])
                          & (self.bev[:, 1] < box_range[3]))
        return in_range_flags

    @abstractmethod
    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[
            BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        pass

    @abstractmethod
    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        pass

    def translate(self, trans_vector: Union[Tensor, np.ndarray]) -> None:
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size
                1x3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(
            self, box_range: Union[Tensor, np.ndarray,
                                   Sequence[float]]) -> Tensor:
        """Check whether the boxes are in the given range.

        Args:
            box_range (Tensor or np.ndarray or Sequence[float]): The range of
                box (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        gravity_center = self.gravity_center
        in_range_flags = ((gravity_center[:, 0] > box_range[0])
                          & (gravity_center[:, 1] > box_range[1])
                          & (gravity_center[:, 2] > box_range[2])
                          & (gravity_center[:, 0] < box_range[3])
                          & (gravity_center[:, 1] < box_range[4])
                          & (gravity_center[:, 2] < box_range[5]))
        return in_range_flags

    @abstractmethod
    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
                   correct_yaw: bool = False) -> 'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        pass

    def scale(self, scale_factor: float) -> None:
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor  # velocity

    def limit_yaw(self, offset: float = 0.5, period: float = np.pi) -> None:
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw. Defaults to 0.5.
            period (float): The expected period. Defaults to np.pi.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold: float = 0.0) -> Tensor:
        """Find boxes that are non-empty.

        A box is considered empty if either of its side is no larger than
        threshold.

        Args:
            threshold (float): The threshold of minimal sizes. Defaults to 0.0.

        Returns:
            Tensor: A binary vector which represents whether each box is empty
            (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = ((size_x > threshold)
                & (size_y > threshold) & (size_z > threshold))
        return keep

    def __getitem__(
            self, item: Union[int, slice, np.ndarray,
                              Tensor]) -> 'BaseInstance3DBoxes':
        """
        Args:
            item (int or slice or np.ndarray or Tensor): Index of boxes.

        Note:
            The following usage are allowed:

            1. `new_boxes = boxes[3]`: Return a `Boxes` that contains only one
               box.
            2. `new_boxes = boxes[2:10]`: Return a slice of boxes.
            3. `new_boxes = boxes[vector]`: Where vector is a
               torch.BoolTensor with `length = len(boxes)`. Nonzero elements in
               the vector will be selected.

            Note that the returned Boxes might share storage with this Boxes,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
            :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                box_dim=self.box_dim,
                with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self) -> int:
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list: Sequence['BaseInstance3DBoxes']
            ) -> 'BaseInstance3DBoxes':
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (Sequence[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned boxes never share storage with input
        cat_boxes = cls(
            torch.cat([b.tensor for b in boxes_list], dim=0),
            box_dim=boxes_list[0].box_dim,
            with_yaw=boxes_list[0].with_yaw)
        return cat_boxes

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: Union[str, torch.device], *args,
           **kwargs) -> 'BaseInstance3DBoxes':
        """Convert current boxes to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the specific
            device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device, *args, **kwargs),
            box_dim=self.box_dim,
            with_yaw=self.with_yaw)

    def cpu(self) -> 'BaseInstance3DBoxes':
        """Convert current boxes to cpu device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cpu device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cpu(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def cuda(self, *args, **kwargs) -> 'BaseInstance3DBoxes':
        """Convert current boxes to cuda device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cuda device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cuda(*args, **kwargs),
            box_dim=self.box_dim,
            with_yaw=self.with_yaw)

    def clone(self) -> 'BaseInstance3DBoxes':
        """Clone the boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties as
            self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def detach(self) -> 'BaseInstance3DBoxes':
        """Detach the boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties as
            self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.detach(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def device(self) -> torch.device:
        """torch.device: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self) -> Iterator[Tensor]:
        """Yield a box as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A box of shape (box_dim, ).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1: 'BaseInstance3DBoxes',
                        boxes2: 'BaseInstance3DBoxes') -> Tensor:
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.

        Returns:
            Tensor: Calculated height overlap of the boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), \
            '"boxes1" and "boxes2" should be in the same type, ' \
            f'but got {type(boxes1)} and {type(boxes2)}.'

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        heighest_of_bottom = torch.max(boxes1_bottom_height,
                                       boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    @classmethod
    def overlaps(cls,
                 boxes1: 'BaseInstance3DBoxes',
                 boxes2: 'BaseInstance3DBoxes',
                 mode: str = 'iou') -> Tensor:
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            Tensor: Calculated 3D overlap of the boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), \
            '"boxes1" and "boxes2" should be in the same type, ' \
            f'but got {type(boxes1)} and {type(boxes2)}.'

        assert mode in ['iou', 'iof']

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        # height overlap
        overlaps_h = cls.height_overlaps(boxes1, boxes2)

        # Restrict the min values of W and H to avoid memory overflow in
        # ``box_iou_rotated``.
        boxes1_bev, boxes2_bev = boxes1.bev, boxes2.bev
        boxes1_bev[:, 2:4] = boxes1_bev[:, 2:4].clamp(min=1e-4)
        boxes2_bev[:, 2:4] = boxes2_bev[:, 2:4].clamp(min=1e-4)

        # bev overlap
        if MMCV_OPS_AVAILABLE:
            iou2d = box_iou_rotated(boxes1_bev, boxes2_bev)
        else:
            iou2d = _box_iou_rotated_cpu(boxes1_bev, boxes2_bev)
        areas1 = (boxes1_bev[:, 2] * boxes1_bev[:, 3]).unsqueeze(1).expand(
            rows, cols)
        areas2 = (boxes2_bev[:, 2] * boxes2_bev[:, 3]).unsqueeze(0).expand(
            rows, cols)
        overlaps_bev = iou2d * (areas1 + areas2) / (1 + iou2d)

        # 3d overlaps
        overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h

        volume1 = boxes1.volume.view(-1, 1)
        volume2 = boxes2.volume.view(1, -1)

        if mode == 'iou':
            # the clamp func is used to avoid division of 0
            iou3d = overlaps_3d / torch.clamp(
                volume1 + volume2 - overlaps_3d, min=1e-8)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-8)

        return iou3d

    def new_box(
        self, data: Union[Tensor, np.ndarray, Sequence[Sequence[float]]]
    ) -> 'BaseInstance3DBoxes':
        """Create a new box object with data.

        The new box and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``, the
            object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(
            new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def points_in_boxes_part(
            self,
            points: Tensor,
            boxes_override: Optional[Tensor] = None) -> Tensor:
        """Find the box in which each point is.

        Args:
            points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
                are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (Tensor, optional): Boxes to override `self.tensor`.
                Defaults to None.

        Note:
            If a point is enclosed by multiple boxes, the index of the first
            box will be returned.

        Returns:
            Tensor: The index of the first box that each point is in with shape
            (M, ). Default value is -1 (if the point is not enclosed by any
            box).
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor

        points_clone = points.clone()[..., :3]
        if points_clone.dim() == 2:
            points_clone = points_clone.unsqueeze(0)
        else:
            assert points_clone.dim() == 3 and points_clone.shape[0] == 1

        boxes = boxes.to(points_clone.device)
        if MMCV_OPS_AVAILABLE:
            box_idx = points_in_boxes_part(points_clone, boxes.unsqueeze(0))
            return box_idx.squeeze(0)

        point_cloud = points_clone.squeeze(0)
        box_corners = type(self)(
            boxes, box_dim=self.box_dim, with_yaw=self.with_yaw).corners
        box_idx = point_cloud.new_full((point_cloud.shape[0], ),
                                       -1,
                                       dtype=torch.long)
        for idx, corners in enumerate(box_corners):
            inside = _points_in_single_box(point_cloud, corners)
            box_idx[(box_idx < 0) & inside] = idx
        return box_idx

    def points_in_boxes_all(self,
                            points: Tensor,
                            boxes_override: Optional[Tensor] = None) -> Tensor:
        """Find all boxes in which each point is.

        Args:
            points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
                are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (Tensor, optional): Boxes to override `self.tensor`.
                Defaults to None.

        Returns:
            Tensor: A tensor indicating whether a point is in a box with shape
            (M, T). T is the number of boxes. Denote this tensor as A, it the
            m^th point is in the t^th box, then `A[m, t] == 1`, otherwise
            `A[m, t] == 0`.
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor

        points_clone = points.clone()[..., :3]
        if points_clone.dim() == 2:
            points_clone = points_clone.unsqueeze(0)
        else:
            assert points_clone.dim() == 3 and points_clone.shape[0] == 1

        boxes = boxes.to(points_clone.device)
        if MMCV_OPS_AVAILABLE:
            box_idxs_of_pts = points_in_boxes_all(points_clone,
                                                  boxes.unsqueeze(0))
            return box_idxs_of_pts.squeeze(0)

        point_cloud = points_clone.squeeze(0)
        box_corners = type(self)(
            boxes, box_dim=self.box_dim, with_yaw=self.with_yaw).corners
        box_idxs_of_pts = point_cloud.new_zeros((point_cloud.shape[0],
                                                 box_corners.shape[0]),
                                                dtype=torch.int32)
        for idx, corners in enumerate(box_corners):
            box_idxs_of_pts[:, idx] = _points_in_single_box(
                point_cloud, corners).to(box_idxs_of_pts.dtype)
        return box_idxs_of_pts

    def points_in_boxes(self,
                        points: Tensor,
                        boxes_override: Optional[Tensor] = None) -> Tensor:
        warnings.warn('DeprecationWarning: points_in_boxes is a deprecated '
                      'method, please consider using points_in_boxes_part.')
        return self.points_in_boxes_part(points, boxes_override)

    def points_in_boxes_batch(
            self,
            points: Tensor,
            boxes_override: Optional[Tensor] = None) -> Tensor:
        warnings.warn('DeprecationWarning: points_in_boxes_batch is a '
                      'deprecated method, please consider using '
                      'points_in_boxes_all.')
        return self.points_in_boxes_all(points, boxes_override)
