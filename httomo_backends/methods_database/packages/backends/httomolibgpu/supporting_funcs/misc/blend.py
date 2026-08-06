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
"""Modules supporting blend functions"""

from typing import Tuple

__all__ = [
    "_calc_output_dim_seam_blend_stitched_data",
]


def _calc_output_dim_seam_blend_stitched_data(
    non_slice_dims_shape: Tuple[int, int],
    **kwargs,
) -> Tuple[int, int]:
    overlap: int = kwargs["overlap"]

    return non_slice_dims_shape[0], non_slice_dims_shape[1] - overlap
