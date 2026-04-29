# Copyright (c) OpenMMLab. All rights reserved.
from .._import_utils import optional_import

__all__ = []

try:
    from mmdet.models.losses import FocalLoss, SmoothL1Loss, binary_cross_entropy
    __all__ += ['FocalLoss', 'SmoothL1Loss', 'binary_cross_entropy']
except Exception:
    pass

optional_import('.cross_entropy_loss', ['CrossEntropyLoss'], globals(),
                __all__)

optional_import('.axis_aligned_iou_loss',
                ['AxisAlignedIoULoss', 'axis_aligned_iou_loss'], globals(),
                __all__)
optional_import('.chamfer_distance', ['ChamferDistance', 'chamfer_distance'],
                globals(), __all__)
optional_import('.lovasz_loss', ['LovaszLoss'], globals(), __all__)
optional_import('.multibin_loss', ['MultiBinLoss'], globals(), __all__)
optional_import('.paconv_regularization_loss', ['PAConvRegularizationLoss'],
                globals(), __all__)
optional_import('.rotated_iou_loss',
                ['RotatedIoU3DLoss', 'rotated_iou_3d_loss'], globals(),
                __all__)
optional_import('.uncertain_smooth_l1_loss',
                ['UncertainL1Loss', 'UncertainSmoothL1Loss'], globals(),
                __all__)
