from . import cupy
from . import cupyx

__all__ = ["cupy", "cupyx"]


def override_globals_shim(function_to_call, globals_source, *args, **kwargs):
    if hasattr(function_to_call, "__globals__"):
        function_to_call.__globals__.update(globals_source.__globals__)

    return function_to_call(*args, **kwargs)
