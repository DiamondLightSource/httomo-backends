import inspect
import math
from typing import Tuple

import numpy as np

from httomolibgpu.misc.denoise import non_local_means

__all__ = [
    "_calc_padding_total_variation_ROF",
    "_calc_memory_bytes_total_variation_ROF",
    "_calc_padding_total_variation_PD",
    "_calc_memory_bytes_total_variation_PD",
    "_calc_padding_total_variation_SB",
    "_calc_memory_bytes_total_variation_SB",
    "_calc_padding_total_variation_FGP",
    "_calc_memory_bytes_total_variation_FGP",
    "_calc_padding_total_generalised_variation",
    "_calc_memory_bytes_total_generalised_variation",
    "_calc_padding_nonlinear_diffusion",
    "_calc_memory_bytes_nonlinear_diffusion",
    "_calc_padding_non_local_means",
    "_calc_memory_bytes_non_local_means",
]


def _calc_padding_total_variation_ROF(**kwargs) -> Tuple[int, int]:
    return (5, 5)


def _calc_padding_total_variation_PD(**kwargs) -> Tuple[int, int]:
    return (5, 5)


def _calc_padding_total_variation_SB(**kwargs) -> Tuple[int, int]:
    """support for approximate slab processing."""
    return (5, 5)


def _calc_padding_total_variation_FGP(**kwargs) -> Tuple[int, int]:
    """Use the existing TV five-slice overlap policy for approximate slab processing."""
    return (5, 5)


def _calc_padding_total_generalised_variation(**kwargs) -> Tuple[int, int]:
    """Use the existing TV five-slice overlap policy for approximate slab processing."""
    return (5, 5)


def _calc_padding_nonlinear_diffusion(**kwargs) -> Tuple[int, int]:
    """Use the existing five-slice overlap policy for approximate slab processing."""
    return (5, 5)


def _calc_memory_bytes_nonlinear_diffusion(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    """Reserve input, packing, two image buffers and a boolean finite mask.

    NDF supports float32 only. Reserve packing even for C-contiguous input.
    Fixed overhead covers rounding of the arrays and small reduction buffers;
    CUDA context/kernel compilation memory is outside this array-memory estimate.
    """
    if kwargs.get("half_precision", False):
        raise ValueError("nonlinear_diffusion requires half_precision=False")
    pixels = math.prod(int(size) for size in non_slice_dims_shape)
    bytes_per_pixel = (
        np.dtype(dtype).itemsize
        + 3 * np.dtype(np.float32).itemsize
        + np.dtype(np.bool_).itemsize
    )
    return pixels * bytes_per_pixel, 6 * 512


def _calc_memory_bytes_total_generalised_variation(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    """Reserve input, packing, primal/dual storage and finite-check workspace.

    The 3D solver holds three scalar primal volumes and three vector primal
    buffers (nine volume components), all float32. Its vector and symmetric
    tensor dual fields have nine components in total; only these use half
    precision. Reserve another float32 volume for the three-component boolean
    finite mask and reduction workspace. Keep this 3D bound for single slices
    and reserve packing even for C-contiguous input. Fixed overhead covers
    allocation rounding and small reduction buffers. CUDA context/kernel
    compilation memory is outside this array-memory estimate.
    """
    pixels = math.prod(int(size) for size in non_slice_dims_shape)
    auxiliary_dtype = np.float16 if kwargs.get("half_precision", False) else np.float32
    bytes_per_pixel = (
        np.dtype(dtype).itemsize
        + 14 * np.dtype(np.float32).itemsize
        + 9 * np.dtype(auxiliary_dtype).itemsize
    )
    return pixels * bytes_per_pixel, 24 * 512


def _calc_memory_bytes_total_variation_FGP(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    """Reserve input, a packed copy, one image and twelve auxiliary buffers.

    Use the 3D buffer count also for single-slice processing, since slab depth
    is unknown here. Packing is reserved even for C-contiguous input. The four
    sets of three dual fields use half precision when requested; the image
    remains float32. Fixed overhead covers pool rounding of the fifteen buffers
    to 512-byte allocation units; CUDA context/kernel compilation memory is
    outside this array-memory estimate.
    """
    pixels = math.prod(int(size) for size in non_slice_dims_shape)
    auxiliary_dtype = np.float16 if kwargs.get("half_precision", False) else np.float32
    bytes_per_pixel = (
        np.dtype(dtype).itemsize
        + 2 * np.dtype(np.float32).itemsize
        + 12 * np.dtype(auxiliary_dtype).itemsize
    )
    return pixels * bytes_per_pixel, 15 * 512


def _calc_memory_bytes_total_variation_SB(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    """Reserve input, a packed copy, two image and six split/Bregman buffers.

    Use the 3D buffer count also for single-slice processing, since slab depth
    is unknown here. Packing is reserved even for C-contiguous input. Only the
    split/Bregman buffers use half precision. Fixed overhead covers pool rounding
    of the ten buffers to 512-byte allocation units; CUDA context/kernel
    compilation memory is outside this array-memory estimate.
    """
    pixels = math.prod(int(size) for size in non_slice_dims_shape)
    auxiliary_dtype = np.float16 if kwargs.get("half_precision", False) else np.float32
    bytes_per_pixel = (
        np.dtype(dtype).itemsize
        + 3 * np.dtype(np.float32).itemsize
        + 6 * np.dtype(auxiliary_dtype).itemsize
    )
    return pixels * bytes_per_pixel, 10 * 512


def _calc_memory_bytes_total_variation_ROF(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    """Reserve input, a packed copy, two image and three auxiliary buffers.

    Use the 3D buffer count also for single-slice processing, since slab depth
    is unknown here. Packing is reserved even for C-contiguous input. Only the
    auxiliary buffers use half precision. Fixed overhead covers pool rounding
    of the seven buffers to 512-byte allocation units; CUDA context/kernel
    compilation memory is outside this array-memory estimate.
    """
    pixels = math.prod(int(size) for size in non_slice_dims_shape)
    auxiliary_dtype = np.float16 if kwargs.get("half_precision", False) else np.float32
    bytes_per_pixel = (
        np.dtype(dtype).itemsize
        + 3 * np.dtype(np.float32).itemsize
        + 3 * np.dtype(auxiliary_dtype).itemsize
    )
    return pixels * bytes_per_pixel, 7 * 512


def _calc_memory_bytes_total_variation_PD(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    """Reserve input, a packed copy, three primal and six dual buffers.

    Use the 3D buffer count also for single-slice processing, since slab depth
    is unknown here. Packing is reserved even for C-contiguous input. Only the
    dual buffers use half precision. Fixed overhead covers pool rounding of
    the eleven buffers to 512-byte allocation units; CUDA context/kernel
    compilation memory is outside this array-memory estimate.
    """
    pixels = math.prod(int(size) for size in non_slice_dims_shape)
    dual_dtype = np.float16 if kwargs.get("half_precision", False) else np.float32
    bytes_per_pixel = (
        np.dtype(dtype).itemsize
        + 4 * np.dtype(np.float32).itemsize
        + 6 * np.dtype(dual_dtype).itemsize
    )
    return pixels * bytes_per_pixel, 11 * 512


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

    Reserve input, output and a C-contiguous input copy (needed for non-C-contiguous
    input). There is no volume-sized patch array or neighbour graph.
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
