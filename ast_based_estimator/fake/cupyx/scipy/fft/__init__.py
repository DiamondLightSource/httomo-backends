import numpy as np
from ....cupy import ndarray

from httomo_backends.cufft import CufftType, cufft_estimate_1d, cufft_estimate_2d

def _output_dtype(dtype, value_type):
    if value_type != 'R2C':
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


def fft():
def ifft2():
def rfftfreq():
def rfft(array, axis, **kwargs):
    array = _convert_dtype(array, "R2C")
    shape = array.shape
    shape[axis] = shape[axis] // 2 + 1

    fft_plan_cache = kwargs.get("fft_plan_cache")
    if fft_plan_cache:
        plan_size = cufft_estimate_1d(nx=shape[axis], fft_type=CufftType.CUFFT_R2C, batch=shape[axis - 1])
        fft_plan_cache.append(ndarray(plan_size, np.int8, kwargs))

    return ndarray(shape, array.dtype, kwargs)


def irfft():