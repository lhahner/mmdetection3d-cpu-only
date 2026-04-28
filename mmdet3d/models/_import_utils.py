# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import warnings
from typing import Dict, Iterable, List


def optional_import(module_name: str,
                    names: Iterable[str],
                    namespace: Dict[str, object],
                    exported: List[str]) -> None:
    """Import optional modules without failing the whole package import."""
    try:
        module = importlib.import_module(module_name, package=__package__)
    except Exception as exc:
        warnings.warn(f'Skipping optional import {module_name}: {exc}')
        return

    for name in names:
        namespace[name] = getattr(module, name)
    exported.extend(names)
