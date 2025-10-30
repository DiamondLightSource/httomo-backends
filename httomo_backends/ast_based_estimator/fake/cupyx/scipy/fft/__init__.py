import numpy as np
from .... import fft_plan_cache
from ....cupy import ndarray

from httomo_backends.cufft import CufftType, cufft_estimate_1d, cufft_estimate_2d


def _output_dtype(dtype, value_type):
    if value_type != "R2C":
        if dtype in [np.float16, np.float32]:
            return np.complex64
        elif dtype not in [np.complex64, np.complex128]:
            return np.complex128
    else:
        if dtype in [np.complex64, np.complex128]:
            return np.dtype(dtype.char.lower())
        elif dtype == np.float16:
            return np.float32
        elif dtype not in [np.float32, np.float64]:
            return np.float64
    return dtype


def _convert_dtype(a, value_type):
    out_dtype = _output_dtype(a.dtype, value_type)
    if out_dtype != a.dtype:
        a = a.astype(out_dtype)
    return a


def fft(array):
    array = _convert_dtype(array, "C2C")
    shape = array.shape

    if fft_plan_cache:
        plan_key = (shape, CufftType.CUFFT_C2C, array.size // array.shape[-1])
        cached_plan = fft_plan_cache.get(plan_key)
        if cached_plan is None:
            plan_size = cufft_estimate_1d(
                nx=shape[-1], fft_type=CufftType.CUFFT_C2C, batch=shape[-2]
            )
            fft_plan_cache[plan_key] = ndarray(plan_size, np.byte)

    return ndarray(shape, array.dtype)


def rfft(array, axis=-1):
    array = _convert_dtype(array, "R2C")
    shape = list(array.shape)
    shape[axis] = shape[axis] // 2 + 1
    shape = tuple(shape)

    if fft_plan_cache:
        plan_key = (shape, CufftType.CUFFT_R2C, array.size // array.shape[-1])
        cached_plan = fft_plan_cache.get(plan_key)
        if cached_plan is None:
            plan_size = cufft_estimate_1d(
                nx=shape[axis], fft_type=CufftType.CUFFT_R2C, batch=shape[axis - 1]
            )
            fft_plan_cache[plan_key] = ndarray(plan_size, np.byte)

    return ndarray(shape, array.dtype)


def irfft(array, axis=-1):
    array = _convert_dtype(array, "C2R")
    shape = list(array.shape)
    shape[axis] = 2 * (shape[axis] + 1)
    shape = tuple(shape)

    if fft_plan_cache:
        plan_key = (shape, CufftType.CUFFT_C2R, array.size // array.shape[-1])
        cached_plan = fft_plan_cache.get(plan_key)
        if cached_plan is None:
            plan_size = cufft_estimate_1d(
                nx=shape[axis], fft_type=CufftType.CUFFT_C2R, batch=shape[axis - 1]
            )
            fft_plan_cache[plan_key] = ndarray(plan_size, np.byte)

    return ndarray(shape, array.dtype)


def ifft2(array, axes, overwrite_x):
    array = _convert_dtype(array, "C2C")
    shape = array.shape

    if fft_plan_cache:
        plan_key = (shape, CufftType.CUFFT_C2C, array.size // array.shape[-1])
        cached_plan = fft_plan_cache.get(plan_key)
        if cached_plan is None:
            plan_size = cufft_estimate_2d(
                nx=shape[-1], ny=shape[-2], fft_type=CufftType.CUFFT_C2C
            )
            fft_plan_cache[plan_key] = ndarray(plan_size, np.byte)

    # if overwrite_x:
    #     return array

    return ndarray(shape, array.dtype)


def rfftfreq(n):
    return ndarray(n // 2 + 1, dtype=np.float64) / n
