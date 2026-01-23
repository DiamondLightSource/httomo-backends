import inspect
from typing import Tuple

from httomolibgpu.misc.corr import median_filter, remove_outlier

__all__ = [
    "_calc_padding_remove_outlier",
    "_calc_padding_median_filter",
]


def _calc_padding_remove_outlier(**kwargs) -> Tuple[int, int]:
    if "kernel_size" not in kwargs:
        params = inspect.signature(remove_outlier).parameters
        kwargs["kernel_size"] = params["kernel_size"].default

    kernel_size = kwargs["kernel_size"]
    return (kernel_size // 2, kernel_size // 2)


def _calc_padding_median_filter(**kwargs) -> Tuple[int, int]:
    if "kernel_size" not in kwargs:
        params = inspect.signature(median_filter).parameters
        kwargs["kernel_size"] = params["kernel_size"].default

    kernel_size = kwargs["kernel_size"]
    return (kernel_size // 2, kernel_size // 2)
