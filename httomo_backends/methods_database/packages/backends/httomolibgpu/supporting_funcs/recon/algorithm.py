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
"""Modules for memory estimation for reconstruction algorithms"""

import math
from typing import Tuple
import numpy as np
from httomo_backends.cufft import CufftType, cufft_estimate_1d
from httomolibgpu.recon.algorithm import LPRec3d_tomobar

__all__ = [
    "_calc_memory_bytes_FBP3d_tomobar",
    "_calc_memory_bytes_for_slices_LPRec3d_tomobar",
    "_calc_memory_bytes_SIRT3d_tomobar",
    "_calc_memory_bytes_CGLS3d_tomobar",
    "_calc_memory_bytes_FISTA3d_tomobar",
    "_calc_output_dim_FBP2d_astra",
    "_calc_output_dim_FBP3d_tomobar",
    "_calc_output_dim_LPRec3d_tomobar",
    "_calc_output_dim_SIRT3d_tomobar",
    "_calc_output_dim_CGLS3d_tomobar",
    "_calc_output_dim_FISTA3d_tomobar",
    "_calc_padding_FISTA3d_tomobar",
]


def _calc_padding_FISTA3d_tomobar(**kwargs) -> Tuple[int, int]:
    return (5, 5)


def __calc_output_dim_recon(non_slice_dims_shape, **kwargs):
    """Function to calculate output dimensions for all reconstructors.
    The change of the dimension depends either on the user-provided "recon_size"
    parameter or taken as the size of the horizontal detector (default).

    """
    DetectorsLengthH = non_slice_dims_shape[1]
    recon_size = kwargs["recon_size"]
    if recon_size is None:
        recon_size = DetectorsLengthH
    output_dims = (recon_size, recon_size)
    return output_dims


def _calc_output_dim_FBP2d_astra(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_FBP3d_tomobar(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_LPRec3d_tomobar(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_SIRT3d_tomobar(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_CGLS3d_tomobar(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_FISTA3d_tomobar(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_memory_bytes_FBP3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    detector_pad = 0
    if "detector_pad" in kwargs:
        detector_pad = kwargs["detector_pad"]
    if detector_pad is True:
        detector_pad = __estimate_detectorHoriz_padding(non_slice_dims_shape[1])
    elif detector_pad is False:
        detector_pad = 0

    angles_tot = non_slice_dims_shape[0]
    det_width = non_slice_dims_shape[1] + 2 * detector_pad
    SLICES = 200  # dummy multiplier+divisor to pass large batch size threshold

    # 1. input
    input_slice_size = (angles_tot * det_width) * dtype.itemsize

    ########## FFT / filter / IFFT (filtersync_cupy)

    # 2. RFFT plan (R2C transform)
    fftplan_slice_size = (
        cufft_estimate_1d(
            nx=det_width,
            fft_type=CufftType.CUFFT_R2C,
            batch=angles_tot * SLICES,
        )
        / SLICES
    )

    # 3. RFFT output size (proj_f in code)
    proj_f_slice = angles_tot * (det_width // 2 + 1) * np.complex64().itemsize

    # 4. Filter size (independent of number of slices)
    filter_size = (det_width // 2 + 1) * np.float32().itemsize

    # 5. IRFFT plan size
    ifftplan_slice_size = (
        cufft_estimate_1d(
            nx=det_width,
            fft_type=CufftType.CUFFT_C2R,
            batch=angles_tot * SLICES,
        )
        / SLICES
    )

    # 6. output of filtersync call
    filtersync_output_slice_size = input_slice_size

    # since the FFT plans, proj_f, and input data is dropped after the filtersync call, we track it here
    # separate
    filtersync_size = (
        input_slice_size + fftplan_slice_size + proj_f_slice + ifftplan_slice_size
    )

    # 6. we swap the axes before passing data to Astra in ToMoBAR
    # https://github.com/dkazanc/ToMoBAR/blob/54137829b6326406e09f6ef9c95eb35c213838a7/tomobar/methodsDIR_CuPy.py#L135
    pre_astra_input_swapaxis_slice = (angles_tot * det_width) * np.float32().itemsize

    # 7. astra backprojection will generate an output array
    # https://github.com/dkazanc/ToMoBAR/blob/54137829b6326406e09f6ef9c95eb35c213838a7/tomobar/astra_wrappers/astra_base.py#L524
    output_dims = _calc_output_dim_FBP3d_tomobar((angles_tot, det_width), **kwargs)
    recon_output_size = np.prod(output_dims) * np.float32().itemsize

    # 7. astra backprojection makes a copy of the input
    astra_input_slice_size = (angles_tot * det_width) * np.float32().itemsize

    ## now we calculate back projection memory (2 copies of the input + reconstruction output)
    projection_mem_size = (
        pre_astra_input_swapaxis_slice + astra_input_slice_size + recon_output_size
    )

    # 9. apply_circular_mask memory (fixed amount, not per slice)
    circular_mask_size = np.prod(output_dims) * np.int64().itemsize

    fixed_amount = max(filter_size, circular_mask_size)

    # 9. this swapaxis makes another copy of the output data
    # https://github.com/DiamondLightSource/httomolibgpu/blob/72d98ec7ac44e06ee0318043934fb3f68667d203/httomolibgpu/recon/algorithm.py#L118
    # BUT: this swapaxis happens after the cudaArray inputs and the input swapaxis arrays are dropped,
    #      so it does not add to the memory overall

    # We assume for safety here that one FFT plan is not freed and one is freed
    tot_memory_bytes = int(
        projection_mem_size + filtersync_size - ifftplan_slice_size + recon_output_size
    )

    # this account for the memory used for filtration AND backprojection.
    return (tot_memory_bytes, fixed_amount)


def _calc_memory_bytes_for_slices_LPRec3d_tomobar(
    dims_shape: Tuple[int, int, int],
    dtype: np.dtype,
    **kwargs,
) -> int:
    return LPRec3d_tomobar(dims_shape, calc_peak_gpu_mem=True, **kwargs)


def _calc_memory_bytes_SIRT3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    detector_pad = 0
    if "detector_pad" in kwargs:
        detector_pad = kwargs["detector_pad"]
    if detector_pad is True:
        detector_pad = __estimate_detectorHoriz_padding(non_slice_dims_shape[1])
    elif detector_pad is False:
        detector_pad = 0

    anglesnum = non_slice_dims_shape[0]
    DetectorsLengthH_padded = non_slice_dims_shape[1] + 2 * detector_pad
    # calculate the output shape
    output_dims = _calc_output_dim_SIRT3d_tomobar(non_slice_dims_shape, **kwargs)
    recon_data_size_original = (
        np.prod(output_dims) * dtype.itemsize
    )  # x_rec user-defined size

    in_data_size = (anglesnum * DetectorsLengthH_padded) * dtype.itemsize

    output_dims_larger_grid = (DetectorsLengthH_padded, DetectorsLengthH_padded)

    out_data_size = np.prod(output_dims_larger_grid) * dtype.itemsize

    R = in_data_size
    C = out_data_size

    Res = in_data_size
    Res_times_R = Res
    C_times_res = out_data_size

    astra_projection = in_data_size + out_data_size

    tot_memory_bytes = int(
        recon_data_size_original
        + in_data_size
        + out_data_size
        + R
        + C
        + Res
        + Res_times_R
        + C_times_res
        + astra_projection
    )
    return (tot_memory_bytes, 0)


def _calc_memory_bytes_CGLS3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    detector_pad = 0
    if "detector_pad" in kwargs:
        detector_pad = kwargs["detector_pad"]
    if detector_pad is True:
        detector_pad = __estimate_detectorHoriz_padding(non_slice_dims_shape[1])
    elif detector_pad is False:
        detector_pad = 0

    anglesnum = non_slice_dims_shape[0]
    DetectorsLengthH_padded = non_slice_dims_shape[1] + 2 * detector_pad
    # calculate the output shape
    output_dims = _calc_output_dim_CGLS3d_tomobar(non_slice_dims_shape, **kwargs)
    recon_data_size_original = (
        np.prod(output_dims) * dtype.itemsize
    )  # x_rec user-defined size

    in_data_size = (anglesnum * DetectorsLengthH_padded) * dtype.itemsize
    output_dims_larger_grid = (DetectorsLengthH_padded, DetectorsLengthH_padded)
    recon_data_size = (
        np.prod(output_dims_larger_grid) * dtype.itemsize
    )  # large volume in the algorithm
    recon_data_size2 = recon_data_size  # x_rec linearised
    d_recon = recon_data_size
    d_recon2 = d_recon  # linearised, possibly a copy

    data_r = in_data_size
    Ad = recon_data_size
    Ad2 = Ad
    s = data_r
    collection = (
        in_data_size
        + recon_data_size_original
        + recon_data_size
        + recon_data_size2
        + d_recon
        + d_recon2
        + data_r
        + Ad
        + Ad2
        + s
    )
    astra_contribution = in_data_size + recon_data_size

    tot_memory_bytes = int(collection + astra_contribution)
    return (tot_memory_bytes, 0)


def _calc_memory_bytes_FISTA3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    detector_pad = 0
    if "detector_pad" in kwargs:
        detector_pad = kwargs["detector_pad"]
    if detector_pad is True:
        detector_pad = __estimate_detectorHoriz_padding(non_slice_dims_shape[1])
    elif detector_pad is False:
        detector_pad = 0

    anglesnum = non_slice_dims_shape[0]
    DetectorsLengthH_padded = non_slice_dims_shape[1] + 2 * detector_pad

    # calculate the output shape
    output_dims = _calc_output_dim_FISTA3d_tomobar(non_slice_dims_shape, **kwargs)
    recon_data_size_original = (
        np.prod(output_dims) * dtype.itemsize
    )  # recon user-defined size

    in_data_siz_pad = (anglesnum * DetectorsLengthH_padded) * dtype.itemsize
    output_dims_larger_grid = (DetectorsLengthH_padded, DetectorsLengthH_padded)

    residual_grad = in_data_siz_pad
    out_data_size = np.prod(output_dims_larger_grid) * dtype.itemsize
    X_t = out_data_size
    X_old = out_data_size

    grad_fidelity = out_data_size

    fista_part = (
        recon_data_size_original
        + in_data_siz_pad
        + residual_grad
        + grad_fidelity
        + X_t
        + X_old
        + out_data_size
    )
    regul_part = 8 * np.prod(output_dims_larger_grid) * dtype.itemsize

    tot_memory_bytes = int(fista_part + regul_part)
    return (tot_memory_bytes, 0)


def __estimate_detectorHoriz_padding(detX_size) -> int:
    det_half = detX_size // 2
    padded_value_exact = int(np.sqrt(2 * (det_half**2))) - det_half
    padded_add_margin = padded_value_exact // 2
    return padded_value_exact + padded_add_margin
