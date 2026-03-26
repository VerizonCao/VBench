#  hack_registry.py
import logging
import importlib
import os
import sys

# ── Patch mmdet + mmyolo version gates BEFORE they are imported ──────
# mmdet 3.0.0 and mmyolo 0.6.0 ship with mmcv_maximum_version='2.1.0',
# but we use mmcv 2.2.0. Monkey-patch the check at import time.
def _patch_mmcv_version_gate(module_name):
    """Pre-import a module and relax its mmcv version upper bound."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return
        # Read, patch, and exec the __init__.py so the assertion passes
        with open(spec.origin, 'r') as f:
            source = f.read()
        if "mmcv_maximum_version = '2.1.0'" in source:
            source = source.replace(
                "mmcv_maximum_version = '2.1.0'",
                "mmcv_maximum_version = '2.3.0'",
            )
            code = compile(source, spec.origin, 'exec')
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            exec(code, mod.__dict__)
    except Exception:
        pass  # If patching fails, let the normal import path handle it

if 'mmdet' not in sys.modules:
    _patch_mmcv_version_gate('mmdet')
if 'mmyolo' not in sys.modules:
    _patch_mmcv_version_gate('mmyolo')

from mmengine.registry import Registry
from mmengine.logging import print_log
from typing import Type, Optional, Union, List


def _register_module(self,
                     module: Type,
                     module_name: Optional[Union[str, List[str]]] = None,
                     force: bool = False) -> None:
    """Register a module.

    Args:
        module (type): Module to be registered. Typically a class or a
            function, but generally all ``Callable`` are acceptable.
        module_name (str or list of str, optional): The module name to be
            registered. If not specified, the class name will be used.
            Defaults to None.
        force (bool): Whether to override an existing class with the same
            name. Defaults to False.
    """
    if not callable(module):
        raise TypeError(f'module must be Callable, but got {type(module)}')

    if module_name is None:
        module_name = module.__name__
    if isinstance(module_name, str):
        module_name = [module_name]
    for name in module_name:
        if not force and name in self._module_dict:
            existed_module = self.module_dict[name]
            # raise KeyError(f'{name} is already registered in {self.name} '
            #                f'at {existed_module.__module__}')
            print_log(
                f'{name} is already registered in {self.name} '
                f'at {existed_module.__module__}. Registration ignored.',
                logger='current',
                level=logging.INFO
            )
        self._module_dict[name] = module


Registry._register_module = _register_module