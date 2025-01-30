from typing import Tuple


__all__ = [
    "_calc_padding_total_variation_ROF",
    "_calc_padding_total_variation_PD",
]


def _calc_padding_total_variation_ROF(**kwargs) -> Tuple[int, int]:
    return (5, 5)


def _calc_padding_total_variation_PD(**kwargs) -> Tuple[int, int]:
    return (5, 5)
