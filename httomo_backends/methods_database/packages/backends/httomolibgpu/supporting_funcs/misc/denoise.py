import inspect
import math
from typing import Tuple

import numpy as np

from httomolibgpu.misc.denoise import non_local_means

__all__ = [
    "_calc_padding_total_variation_ROF",
    "_calc_padding_total_variation_PD",
    "_calc_padding_non_local_means",
    "_calc_memory_bytes_non_local_means",
]


def _calc_padding_total_variation_ROF(**kwargs) -> Tuple[int, int]:
    return (5, 5)


def _calc_padding_total_variation_PD(**kwargs) -> Tuple[int, int]:
    return (5, 5)


def _calc_padding_non_local_means(**kwargs) -> Tuple[int, int]:
    """Return axial overlap for volumetric or independent-slice NLM."""
    params = inspect.signature(non_local_means).parameters
    if not kwargs.get("full_3d", params["full_3d"].default):
        return (0, 0)
    search_radius = kwargs.get("search_radius", params["search_radius"].default)
    patch_radius = kwargs.get("patch_radius", params["patch_radius"].default)
    overlap = int(search_radius + patch_radius)
    return (overlap, overlap)


def _calc_memory_bytes_non_local_means(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    """Estimate per-slice bytes and fixed overhead for float32 NLM.

    Reserve input, output and a contiguous input copy (needed for strided
    views). There is no volume-sized patch array or neighbour graph.
    Positive num_neighbours uses two float32 arrays per CUDA thread. Their
    placement in registers/local memory depends on the compiler and GPU;
    charging their full size per voxel is a conservative workspace allowance,
    rather than a prediction of CUDA local-memory allocation.

    Fixed overhead covers offsets, Gaussian weights and the temporary arrays
    used to compute those weights. Use the requested patch radius as an upper
    bound: the library may reduce it for small volumes. Round the small
    weight buffers to CuPy's 512-byte memory-pool allocation unit. A 10%
    margin covers additional rounding for typical tomography-sized slices.
    CUDA context and compilation overhead are not included.
    """
    params = inspect.signature(non_local_means).parameters
    search_radius = int(kwargs.get("search_radius", params["search_radius"].default))
    patch_radius = int(kwargs.get("patch_radius", params["patch_radius"].default))
    neighbours = int(kwargs.get("num_neighbours", params["num_neighbours"].default))
    full_3d = kwargs.get("full_3d", params["full_3d"].default)

    # Bound the kernel's selected-neighbour count without knowing slab depth.
    window = 2 * search_radius + 1
    candidates = math.prod(min(int(size), window) for size in non_slice_dims_shape)
    if full_3d:
        candidates *= window
    neighbours = min(neighbours, candidates - 1)

    pixels = math.prod(int(size) for size in non_slice_dims_shape)
    float_bytes = np.dtype(np.float32).itemsize
    arrays = pixels * (np.dtype(dtype).itemsize + 2 * float_bytes)
    selection_workspace = pixels * 2 * neighbours * float_bytes
    weight_bytes = (2 * patch_radius + 1) * float_bytes
    gaussian_workspace = 5 * ((weight_bytes + 511) // 512) * 512
    return (
        math.ceil(1.1 * (arrays + selection_workspace)),
        math.ceil(1.1 * gaussian_workspace),
    )
