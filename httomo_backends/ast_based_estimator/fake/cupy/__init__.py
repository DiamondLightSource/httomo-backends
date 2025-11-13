import math
import numpy as np
import sys

from .. import memory_usage
from . import cuda, fft

__all__ = ["cuda", "fft"]


__name__ = "cupy"


def get_array_module(array):
    return sys.modules[__name__]


class ndarray:
    def __init__(self, shape, dtype="float32"):
        if not isinstance(shape, tuple) and not isinstance(shape, list):
            shape = (shape,)

        self.shape = shape
        self.dtype = dtype

        if memory_usage:
            memory_usage["current_peak_memory"] += (
                np.prod(self.shape) * np.dtype(self.dtype).itemsize
            )
            memory_usage["peak_memory"] = max(
                memory_usage["peak_memory"],
                memory_usage["current_peak_memory"],
            )

        # print(f"[CREATE] ndarray(shape={self.shape}, dtype={self.dtype})")

    def __del__(self):
        if memory_usage:
            memory_usage["current_peak_memory"] -= (
                np.prod(self.shape) * np.dtype(self.dtype).itemsize
            )
            memory_usage["peak_memory"] = max(
                memory_usage["peak_memory"],
                memory_usage["current_peak_memory"],
            )

        # print(f"[DELETE] ndarray(shape={self.shape}, dtype={self.dtype})")

    def __repr__(self):
        return f"ndarray(memory_usage={memory_usage}, shape={self.shape}, dtype={self.dtype})"

    def _binary_op(self, other):
        if isinstance(other, ndarray):
            result_shape = np.broadcast_shapes(self.shape, other.shape)
            result_dtype = np.result_type(self.dtype, other.dtype)
        else:
            result_shape = self.shape
            result_dtype = np.result_type(self.dtype, type(other))

        return ndarray(result_shape, str(result_dtype))

    def __add__(self, other):
        return self._binary_op(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other)

    def __rsub__(self, other):
        return self._binary_op(other)

    def __mul__(self, other):
        return self._binary_op(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other)

    def __rtruediv__(self, other):
        return self._binary_op(other)

    def __pow__(self, other):
        return self._binary_op(other)

    def __le__(self, other):
        return self._binary_op(other)

    def __getitem__(self, key):
        if isinstance(key, ndarray):
            return ndarray(key.shape, self.dtype)

        if not isinstance(key, tuple):
            key = (key,)

        if Ellipsis in key:
            ellipsis_index = key.index(Ellipsis)
            num_missing = len(self.shape) - (len(key) - 1)  # -1 for the ellipsis itself
            key = (
                key[:ellipsis_index]
                + (slice(None),) * num_missing
                + key[ellipsis_index + 1 :]
            )

        if len(key) < len(self.shape):
            key = key + (slice(None),) * (len(self.shape) - len(key))

        new_shape = []
        for dim_size, idx in zip(self.shape, key):
            if idx is None:
                new_shape.append(1)
            elif isinstance(idx, int):
                continue
            elif isinstance(idx, slice):
                idx = slice(
                    int(idx.start) if idx.start is not None else None,
                    int(idx.stop) if idx.stop is not None else None,
                    int(idx.step) if idx.step is not None else None,
                )
                start, stop, step = idx.indices(dim_size)
                length = max(0, (stop - start + (step - 1)) // step)
                new_shape.append(length)
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")

        return ndarray(new_shape, self.dtype)

    def __setitem__(self, key, value):
        pass

    def astype(self, dtype):
        return ndarray(self.shape, dtype)

    def get(self):
        return np.empty(self.shape, self.dtype)


float16 = np.float16
float32 = np.float32
float64 = np.float64

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

complex64 = np.complex64

bool_ = np.bool_

pi = np.pi


newaxis = None


def dtype(self, obj):
    return np.dtype(obj)


def zeros(shape, dtype=float32):
    return ndarray(shape, dtype)


def ones(shape, dtype):
    return ndarray(shape, dtype)


def empty(shape, dtype=float32):
    return ndarray(shape, dtype)


def full(shape, dtype):
    return ndarray(shape, dtype)


def array(obj, dtype=float32):
    array = np.array(obj, dtype)
    return ndarray(array.shape, dtype)


def asarray(array, dtype):
    if isinstance(array, ndarray):
        return array

    return ndarray(array.shape, dtype)


def pad(array, pad_width, mode):
    return ndarray(
        tuple(s + p[0] + p[1] for (s, p) in zip(array.shape, pad_width)),
        array.dtype,
    )


def swapaxes(array, axis1, axis2):
    shape = list(array.shape)
    shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
    return ndarray(tuple(shape), array.dtype)


def require(array, dtype=None, requirements=None):
    if not requirements:
        return ndarray(array.shape, dtype)

    copy = "OWNDATA" in requirements
    if copy:
        return ndarray(array.shape, dtype=dtype)

    return array


def exp(array):
    return ndarray(array.shape, array.dtype)


def mean(array, axis, dtype, out):
    return ndarray(array.shape, array.dtype)


def sqrt(array, dtype=None, out=None):
    return ndarray(array.shape, array.dtype)


def argsort(a, axis=-1, kind=None):
    if axis is None:
        shape = (np.prod(a.shape),)
    else:
        shape = a.shape

    return ndarray(shape, np.intp)


def count_nonzero(array):
    pass


def arange(start, stop=None, step=1, dtype=None):
    if stop is None:
        stop = start
        start = 0

    size = int(np.ceil((stop - start) / step))
    return ndarray(size, dtype)


def indices(dimensions, dtype=int):
    dimensions = tuple(dimensions)
    N = len(dimensions)
    return ndarray((N,) + dimensions, dtype=dtype)


class nd_grid:
    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, key):
        if isinstance(key, slice):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, complex):
                step = abs(step)
                length = int(step)
                if step != 1:
                    step = (key.stop - start) / float(step - 1)
                stop = key.stop + step
                return arange(0, length, 1, float) * step + start
            else:
                return arange(start, stop, step)

        size = []
        typ = int
        for k in range(len(key)):
            step = key[k].step
            start = key[k].start
            if start is None:
                start = 0
            if step is None:
                step = 1
            if isinstance(step, complex):
                size.append(int(abs(step)))
                typ = float
            else:
                size.append(int(math.ceil((key[k].stop - start) / (step * 1.0))))
            if (
                isinstance(step, float)
                or isinstance(start, float)
                or isinstance(key[k].stop, float)
            ):
                typ = float
        if self.sparse:
            nn = [arange(_x, dtype=_t) for _x, _t in zip(size, (typ,) * len(size))]
        else:
            nn = indices(size, typ)
        for k in range(len(size)):
            step = key[k].step
            start = key[k].start
            if start is None:
                start = 0
            if step is None:
                step = 1
            if isinstance(step, complex):
                step = int(abs(step))
                if step != 1:
                    step = (key[k].stop - start) / float(step - 1)
            nn[k] = nn[k] * step + start
        if self.sparse:
            slobj = [newaxis] * len(size)
            for k in range(len(size)):
                slobj[k] = slice(None, None)
                nn[k] = nn[k][tuple(slobj)]
                slobj[k] = newaxis
        return nn

    def __len__(self):
        return 0


mgrid = nd_grid(sparse=False)
ogrid = nd_grid(sparse=True)


class RawKernel:
    def __call__(self, grid, block, args, **kwargs):
        pass


class RawModule:
    def __init__(self, options, code, name_expressions):
        pass

    def get_function(self, name: str):
        return RawKernel()


class MemoryPool:
    def free_all_blocks(self, stream=None):
        pass


_default_memory_pool = MemoryPool()