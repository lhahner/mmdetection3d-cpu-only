# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_sa_module
from .point_fp_module import PointFPModule
from .point_sa_module import PointSAModule, PointSAModuleMSG
from .stack_point_sa_module import StackedSAModuleMSG

__all__ = [
    'build_sa_module', 'PointSAModuleMSG', 'PointSAModule', 'PointFPModule',
    'StackedSAModuleMSG'
]

try:
    from .paconv_sa_module import (PAConvCUDASAModule, PAConvCUDASAModuleMSG,
                                   PAConvSAModule, PAConvSAModuleMSG)
    __all__ += [
        'PAConvSAModule', 'PAConvSAModuleMSG', 'PAConvCUDASAModule',
        'PAConvCUDASAModuleMSG'
    ]
except Exception:
    PAConvSAModule = None
    PAConvSAModuleMSG = None
    PAConvCUDASAModule = None
    PAConvCUDASAModuleMSG = None
