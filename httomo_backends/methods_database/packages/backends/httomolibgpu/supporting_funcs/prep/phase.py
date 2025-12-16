#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Tomography Team at DLS <scientificsoftware@diamond.ac.uk>
# Created Date: 21 September 2023
# ---------------------------------------------------------------------------
"""Modules for memory estimation for phase retrieval and phase-contrast enhancement"""

from typing import Tuple
import numpy as np

from httomolibgpu.prep.phase import paganin_filter, paganin_filter_savu_legacy

__all__ = [
    "_calc_memory_bytes_for_slices_paganin_filter",
    "_calc_memory_bytes_for_slices_paganin_filter_savu_legacy",
]


def _calc_memory_bytes_for_slices_paganin_filter(
    dims_shape: Tuple[int, int, int],
    dtype: np.dtype,
    **kwargs,
) -> int:
    return paganin_filter(dims_shape, calc_peak_gpu_mem=True, **kwargs)


def _calc_memory_bytes_for_slices_paganin_filter_savu_legacy(
    dims_shape: Tuple[int, int, int],
    dtype: np.dtype,
    **kwargs,
) -> int:
    return paganin_filter_savu_legacy(dims_shape, calc_peak_gpu_mem=True, **kwargs)

