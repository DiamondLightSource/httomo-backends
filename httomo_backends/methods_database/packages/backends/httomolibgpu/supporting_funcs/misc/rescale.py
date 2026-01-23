import inspect
from typing import Tuple
import numpy as np

from httomolibgpu.misc.rescale import rescale_to_int

__all__ = [
    "_calc_memory_bytes_rescale_to_int",
]


def _calc_memory_bytes_rescale_to_int(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    if "bits" not in kwargs:
        params = inspect.signature(rescale_to_int).parameters
        kwargs["bits"] = params["bits"].default
    bits: int = kwargs["bits"]
    if bits == 8:
        itemsize = 1
    elif bits == 16:
        itemsize = 2
    else:
        itemsize = 4
    safety_multiplier = 1.1
    return (
        int(
            safety_multiplier
            * ((np.prod(non_slice_dims_shape)) * (dtype.itemsize + itemsize))
        ),
        0,
    )
