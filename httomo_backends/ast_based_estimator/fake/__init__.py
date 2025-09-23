from . import cupy
from . import cupyx
from . import tomobar
# from . import httomolibgpu

# __all__ = ["cupy", "cupyx", "httomolibgpu", "tomobar"]
__all__ = ["cupy", "cupyx", "tomobar"]


def override_globals(function_to_call, globals_source):
    if hasattr(function_to_call, "__globals__"):
        function_to_call.__globals__.update(globals_source.__globals__)
