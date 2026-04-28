# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
import warnings
from typing import Dict, Iterable, List


def optional_import(module_name: str,
                    names: Iterable[str],
                    namespace: Dict[str, object],
                    exported: List[str]) -> None:
    """Import optional modules without failing the whole package import."""
    caller_package = namespace.get('__package__')
    if caller_package is None:
        caller_package = inspect.currentframe().f_back.f_globals.get(
            '__package__')
    try:
        module = importlib.import_module(module_name, package=caller_package)
    except Exception as exc:
        warnings.warn(f'Skipping optional import {module_name}: {exc}')
        return

    for name in names:
        namespace[name] = getattr(module, name)
    exported.extend(names)
