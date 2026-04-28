# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import warnings


def _optional_import(module_name: str) -> None:
    try:
        importlib.import_module(module_name, package=__package__)
    except Exception as exc:
        warnings.warn(f'Skipping optional import {module_name}: {exc}')


_optional_import('.layers.fusion_layers')
_optional_import('.backbones')
_optional_import('.data_preprocessors')
_optional_import('.decode_heads')
_optional_import('.dense_heads')
_optional_import('.detectors')
_optional_import('.layers')
_optional_import('.losses')
_optional_import('.middle_encoders')
_optional_import('.necks')
_optional_import('.roi_heads')
_optional_import('.segmentors')
_optional_import('.test_time_augs')
_optional_import('.utils')
_optional_import('.voxel_encoders')
