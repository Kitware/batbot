"""Public package interface for BatBot.

Importing :mod:`batbot` intentionally avoids importing the scientific and ONNX
stacks.  The larger submodules are loaded only when their APIs are used.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

from batbot._config import QUIET, log
from batbot._version import VERSION, __version__
from batbot.api import batch, example, fetch, parallel_pipeline, pipeline, pipeline_multi_wrapper

version = __version__

__all__ = [
    'QUIET',
    'VERSION',
    '__version__',
    'batch',
    'classifier',
    'example',
    'fetch',
    'log',
    'parallel_pipeline',
    'pipeline',
    'pipeline_multi_wrapper',
    'spectrogram',
    'version',
]


def __getattr__(name: str) -> Any:
    """Lazily expose the two computational subpackages."""
    if name in {'classifier', 'spectrogram'}:
        module: ModuleType = import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
