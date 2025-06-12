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
from httomo_backends.cufft import CufftType, cufft_estimate_1d, cufft_estimate_2d

__all__ = [
    "_calc_memory_bytes_FBP3d_tomobar",
    "_calc_memory_bytes_LPRec3d_tomobar",
    "_calc_memory_bytes_SIRT3d_tomobar",
    "_calc_memory_bytes_CGLS3d_tomobar",
    "_calc_output_dim_FBP2d_astra",
    "_calc_output_dim_FBP3d_tomobar",
    "_calc_output_dim_LPRec3d_tomobar",
    "_calc_output_dim_SIRT3d_tomobar",
    "_calc_output_dim_CGLS3d_tomobar",
]


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


def _calc_memory_bytes_FBP3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    det_height = non_slice_dims_shape[0]
    det_width = non_slice_dims_shape[1]
    SLICES = 200  # dummy multiplier+divisor to pass large batch size threshold

    # 1. input
    input_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize

    ########## FFT / filter / IFFT (filtersync_cupy)

    # 2. RFFT plan (R2C transform)
    fftplan_slice_size = (
        cufft_estimate_1d(
            nx=det_width,
            fft_type=CufftType.CUFFT_R2C,
            batch=det_height * SLICES,
        )
        / SLICES
    )

    # 3. RFFT output size (proj_f in code)
    proj_f_slice = det_height * (det_width // 2 + 1) * np.complex64().itemsize

    # 4. Filter size (independent of number of slices)
    filter_size = (det_width // 2 + 1) * np.float32().itemsize

    # 5. IRFFT plan size
    ifftplan_slice_size = (
        cufft_estimate_1d(
            nx=det_width,
            fft_type=CufftType.CUFFT_C2R,
            batch=det_height * SLICES,
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
    pre_astra_input_swapaxis_slice = (
        np.prod(non_slice_dims_shape) * np.float32().itemsize
    )

    # 7. astra backprojection will generate an output array
    # https://github.com/dkazanc/ToMoBAR/blob/54137829b6326406e09f6ef9c95eb35c213838a7/tomobar/astra_wrappers/astra_base.py#L524
    output_dims = _calc_output_dim_FBP3d_tomobar(non_slice_dims_shape, **kwargs)
    recon_output_size = np.prod(output_dims) * np.float32().itemsize

    # 7. astra backprojection makes a copy of the input
    astra_input_slice_size = np.prod(non_slice_dims_shape) * np.float32().itemsize

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
    tot_memory_bytes = (
        projection_mem_size + filtersync_size - ifftplan_slice_size + recon_output_size
    )

    # this account for the memory used for filtration AND backprojection.
    return (tot_memory_bytes, fixed_amount)



def _calc_memory_bytes_LPRec3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    # Based on: https://github.com/dkazanc/ToMoBAR/pull/112/commits/4704ecdc6ded3dd5ec0583c2008aa104f30a8a39

    angles_tot = non_slice_dims_shape[0]
    DetectorsLengthH = non_slice_dims_shape[1]
    SLICES = 200  # dummy multiplier+divisor to pass large batch size threshold

    n = DetectorsLengthH

    odd_horiz = False
    if (n % 2) != 0:
        n = n + 1  # dealing with the odd horizontal detector size
        odd_horiz = True

    eps = 1e-4  # accuracy of usfft
    mu = -np.log(eps) / (2 * n * n)
    m = int(np.ceil(2 * n * 1 / np.pi * np.sqrt(-mu * np.log(eps) + (mu * n) * (mu * n) / 4)))

    center_size = 6144
    center_size = min(center_size, n * 2 + m * 2)

    oversampling_level = 2  # at least 2 or larger required
    ne = oversampling_level * n
    padding_m = ne // 2 - n // 2

    if "angles" in kwargs:
        angles = kwargs["angles"]
        sorted_theta_cpu = np.sort(angles)
        theta_full_range = abs(sorted_theta_cpu[angles_tot-1] - sorted_theta_cpu[0])
        angle_range_pi_count = 1 + int(np.ceil(theta_full_range / math.pi))
    else:
        angle_range_pi_count = 1 + int(np.ceil(2)) # assume a 2 * PI projection angle range

    output_dims = __calc_output_dim_recon(non_slice_dims_shape, **kwargs)
    if odd_horiz:
        output_dims = tuple(x + 1 for x in output_dims)

    in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    padded_in_slice_size = np.prod(non_slice_dims_shape) * np.float32().itemsize
    theta_size = angles_tot * np.float32().itemsize
    sorted_theta_indices_size = angles_tot * np.int64().itemsize
    sorted_theta_size = angles_tot * np.float32().itemsize
    recon_output_size = (n + 1) * (n + 1) * np.float32().itemsize if odd_horiz else n * n * np.float32().itemsize    # 264
    linspace_size = n * np.float32().itemsize
    meshgrid_size = 2 * n * n * np.float32().itemsize
    phi_size = 6 * n * n * np.float32().itemsize
    angle_range_size = center_size * center_size * 1 + angle_range_pi_count * 2 * np.int32().itemsize
    c1dfftshift_size = n * np.int8().itemsize
    c2dfftshift_slice_size = 4 * n * n * np.int8().itemsize
    filter_size = (n // 2 + 1) * np.float32().itemsize
    rfftfreq_size = filter_size
    scaled_filter_size = filter_size
    tmp_p_input_slice = np.prod(non_slice_dims_shape) * np.float32().itemsize
    padded_tmp_p_input_slice = angles_tot * (n + padding_m * 2) * dtype.itemsize
    rfft_result_size = padded_tmp_p_input_slice
    filtered_rfft_result_size = rfft_result_size
    rfft_plan_slice_size = cufft_estimate_1d(nx=(n + padding_m * 2),fft_type=CufftType.CUFFT_R2C,batch=angles_tot * SLICES) / SLICES
    irfft_result_size = filtered_rfft_result_size
    irfft_scratch_memory_size = filtered_rfft_result_size
    irfft_plan_slice_size = cufft_estimate_1d(nx=(n + padding_m * 2),fft_type=CufftType.CUFFT_C2R,batch=angles_tot * SLICES) / SLICES
    conversion_to_complex_size = np.prod(non_slice_dims_shape) * np.complex64().itemsize / 2
    datac_size = np.prod(non_slice_dims_shape) * np.complex64().itemsize / 2
    fde_size = (2 * m + 2 * n) * (2 * m + 2 * n) * np.complex64().itemsize / 2
    shifted_datac_size = datac_size
    fft_result_size = datac_size
    backshifted_datac_size = datac_size
    scaled_backshifted_datac_size = datac_size
    fft_plan_slice_size = cufft_estimate_1d(nx=n,fft_type=CufftType.CUFFT_C2C,batch=angles_tot * SLICES) / SLICES
    fde_view_size = 4 * n * n * np.complex64().itemsize / 2
    shifted_fde_view_size = fde_view_size
    ifft2_scratch_memory_size = fde_view_size
    ifft2_plan_slice_size = cufft_estimate_2d(nx=(2 * n),ny=(2 * n),fft_type=CufftType.CUFFT_C2C) / 2
    fde2_size = n * n * np.complex64().itemsize / 2
    concatenate_size = fde2_size
    circular_mask_size = np.prod(output_dims) / 2 * np.int64().itemsize * 4

    after_recon_swapaxis_slice = np.prod(non_slice_dims_shape) * np.float32().itemsize

    tot_memory_bytes = int(
        max(
            in_slice_size + padded_in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + tmp_p_input_slice + padded_tmp_p_input_slice + rfft_result_size + filtered_rfft_result_size + irfft_result_size + irfft_scratch_memory_size
            , in_slice_size + padded_in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + tmp_p_input_slice + datac_size + conversion_to_complex_size
            , in_slice_size + padded_in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fft_plan_slice_size + fde_size + datac_size + shifted_datac_size + fft_result_size + backshifted_datac_size + scaled_backshifted_datac_size
            , in_slice_size + padded_in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fft_plan_slice_size + ifft2_plan_slice_size + shifted_fde_view_size + ifft2_scratch_memory_size
            , in_slice_size + padded_in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fft_plan_slice_size + ifft2_plan_slice_size + fde2_size + concatenate_size
            , in_slice_size + padded_in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fft_plan_slice_size + ifft2_plan_slice_size + after_recon_swapaxis_slice
        )
    )

    fixed_amount = int(
        max(
            theta_size + phi_size + linspace_size + meshgrid_size
            , theta_size + sorted_theta_indices_size + sorted_theta_size + phi_size + angle_range_size + c1dfftshift_size + c2dfftshift_slice_size + filter_size + rfftfreq_size + scaled_filter_size
            , theta_size + sorted_theta_indices_size + sorted_theta_size + phi_size + circular_mask_size
        )
    )

    return (tot_memory_bytes * 1.1, fixed_amount)



def _calc_memory_bytes_SIRT3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    DetectorsLengthH = non_slice_dims_shape[1]
    # calculate the output shape
    output_dims = _calc_output_dim_SIRT3d_tomobar(non_slice_dims_shape, **kwargs)

    in_data_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_data_size = np.prod(output_dims) * dtype.itemsize

    astra_projection = 2.5 * (in_data_size + out_data_size)

    tot_memory_bytes = 2 * in_data_size + 2 * out_data_size + astra_projection
    return (tot_memory_bytes, 0)


def _calc_memory_bytes_CGLS3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    DetectorsLengthH = non_slice_dims_shape[1]
    # calculate the output shape
    output_dims = _calc_output_dim_CGLS3d_tomobar(non_slice_dims_shape, **kwargs)

    in_data_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_data_size = np.prod(output_dims) * dtype.itemsize

    astra_projection = 2.5 * (in_data_size + out_data_size)

    tot_memory_bytes = 2 * in_data_size + 2 * out_data_size + astra_projection
    return (tot_memory_bytes, 0)
