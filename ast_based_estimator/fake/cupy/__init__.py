import numpy as np


class ndarray:
    def __init__(self, shape, dtype="float32", **kwargs):
        line_number = kwargs.get("line_number")
        memory_usage = kwargs.get("memory_usage")

        self.line_number_of_creation = line_number
        self.shape = shape
        self.dtype = dtype
        self.memory_usage = memory_usage

        if self.memory_usage:
            self.memory_usage["current_peak_memory"] += (
                np.prod(self.shape) * np.dtype(self.dtype).itemsize
            )
            self.memory_usage["peak_memory"] = max(
                self.memory_usage["peak_memory"],
                self.memory_usage["current_peak_memory"],
            )

        print(
            f"[CREATE] ndarray(line_no={self.line_number_of_creation}, shape={self.shape}, dtype={self.dtype})"
        )

    def __del__(self):
        if self.memory_usage:
            self.memory_usage["current_peak_memory"] -= (
                np.prod(self.shape) * np.dtype(self.dtype).itemsize
            )
            self.memory_usage["peak_memory"] = max(
                self.memory_usage["peak_memory"],
                self.memory_usage["current_peak_memory"],
            )

        print(
            f"[DELETE] ndarray(line_no={self.line_number_of_creation}, shape={self.shape}, dtype={self.dtype})"
        )

    def __repr__(self):
        return f"ndarray(line_no={self.line_number_of_creation}, memory_usage={self.memory_usage}, shape={self.shape}, dtype={self.dtype})"

    def _binary_op(self, other):
        if isinstance(other, ndarray):
            line_no = other.line_number_of_creation
            result_shape = np.broadcast_shapes(self.shape, other.shape)
            result_dtype = np.result_type(self.dtype, other.dtype)
        else:
            line_no = self.line_number_of_creation
            result_shape = self.shape
            result_dtype = np.result_type(self.dtype, type(other))

        return ndarray(
            result_shape,
            str(result_dtype),
            **{
                "line_number": line_no,
                "memory_usage": self.memory_usage,
            },
        )

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

    def __getitem__(self, key):
        if isinstance(key, ndarray):
            return ndarray(
                key.shape,
                self.dtype,
                **{
                    "line_number": self.line_number_of_creation,
                    "memory_usage": self.memory_usage,
                },
            )

        if not isinstance(key, tuple):
            key = (key,)

        key = key + (slice(None),) * (len(self.shape) - len(key))

        new_shape = []
        for dim_size, idx in zip(self.shape, key):
            if isinstance(idx, int):
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

        return ndarray(
            new_shape,
            self.dtype,
            **{
                "line_number": self.line_number_of_creation,
                "memory_usage": self.memory_usage,
            },
        )

    def __setitem__(self, key, value):
        pass

    def astype(self, dtype):
        return ndarray(
            self.shape,
            dtype,
            **{
                "line_number": self.line_number_of_creation,
                "memory_usage": self.memory_usage,
            },
        )

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

def dtype(self, obj):
    return np.dtype(obj)


def zeros(shape, dtype, **kwargs):
    return ndarray(shape, dtype, **kwargs)


def ones(shape, dtype, **kwargs):
    return ndarray(shape, dtype, **kwargs)


def empty(shape, dtype, **kwargs):
    return ndarray(shape, dtype, **kwargs)


def full(shape, dtype, **kwargs):
    return ndarray(shape, dtype, **kwargs)


def array(obj, dtype, **kwargs):
    array = np.array(obj, dtype)
    return ndarray(array.shape, dtype, **kwargs)


def asarray(array, dtype, **kwargs):
    if isinstance(array, ndarray):
        return array

    return ndarray(array.shape, dtype, **kwargs)


def pad(array, pad_width, **kwargs):
    return ndarray(
        tuple(s + p[0] + p[1] for (s, p) in zip(array.shape, pad_width)),
        array.dtype,
        **kwargs,
    )


def exp(array, **kwargs):
    return ndarray(array.shape, array.dtype, **kwargs)


def argsort(a, axis=-1, kind=None, **kwargs):
    if axis is None:
        shape = (np.prod(a.shape),)
    else:
        shape = a.shape

    return ndarray(shape, np.intp, **kwargs)
