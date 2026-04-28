# Copyright (c) OpenMMLab. All rights reserved.
from .inference import (convert_SyncBN, inference_detector,
                        inference_mono_3d_detector,
                        inference_multi_modality_detector, inference_segmentor,
                        init_model)

__all__ = [
    'inference_detector', 'init_model', 'inference_mono_3d_detector',
    'convert_SyncBN', 'inference_multi_modality_detector',
    'inference_segmentor', 'Base3DInferencer', 'MonoDet3DInferencer',
    'LidarDet3DInferencer', 'LidarSeg3DInferencer',
    'MultiModalityDet3DInferencer'
]


def __getattr__(name):
    if name in {
            'Base3DInferencer', 'LidarDet3DInferencer',
            'LidarSeg3DInferencer', 'MonoDet3DInferencer',
            'MultiModalityDet3DInferencer'
    }:
        from .inferencers import (Base3DInferencer, LidarDet3DInferencer,
                                  LidarSeg3DInferencer, MonoDet3DInferencer,
                                  MultiModalityDet3DInferencer)
        return {
            'Base3DInferencer': Base3DInferencer,
            'LidarDet3DInferencer': LidarDet3DInferencer,
            'LidarSeg3DInferencer': LidarSeg3DInferencer,
            'MonoDet3DInferencer': MonoDet3DInferencer,
            'MultiModalityDet3DInferencer': MultiModalityDet3DInferencer
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
