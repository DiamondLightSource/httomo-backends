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
    ACTUAL_SLICE_COUNT = 10

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
        angle_range_pi_count += 1 # account for difference from actual algorithm
    else:
        angle_range_pi_count = 1 + int(np.ceil(2)) # assume a 2 * PI projection angle range

    if "chunk_count" in kwargs:
        chunk_count = kwargs["chunk_count"]
    else:
        chunk_count = 2

    output_dims = __calc_output_dim_recon(non_slice_dims_shape, **kwargs)
    if odd_horiz:
        output_dims = tuple(x + 1 for x in output_dims)

    def debug_print(ln: int, name: str, size: int, per_slice: bool):
        slice_multiplier = ACTUAL_SLICE_COUNT if per_slice else 1
        size_kb = size / 1024
        size_mb = size_kb / 1024
        if size_mb < 1:
            print(f"{ln} {name} {size_kb * slice_multiplier} KB")
        else:
            print(f"{ln} {name} {size_mb * slice_multiplier} MB")

    in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    debug_print(-1, "in_slice_size", in_slice_size, True)
    padded_in_slice_size = angles_tot * n * np.float32().itemsize
    debug_print(233, "padded_in_slice_size", padded_in_slice_size, True)

    theta_size = angles_tot * np.float32().itemsize
    debug_print(240, "theta", theta_size, False)
    filter_size = (n // 2 + 1) * np.float32().itemsize
    debug_print(259, "filter_size", filter_size, False)
    rfftfreq_size = filter_size
    debug_print(262, "rfftfreq_size", rfftfreq_size, False)
    scaled_filter_size = filter_size
    debug_print(263, "scaled_filter_size", scaled_filter_size, False)

    tmp_p_input_slice = angles_tot * n * np.float32().itemsize
    debug_print(266, "tmp_p_input", tmp_p_input_slice, True)

    padded_tmp_p_input_slice = angles_tot * (n + padding_m * 2) * dtype.itemsize / chunk_count
    debug_print(273, "padded_tmp_p_input_slice", padded_tmp_p_input_slice, True)
    rfft_plan_slice_size = cufft_estimate_1d(nx=(n + padding_m * 2),fft_type=CufftType.CUFFT_R2C,batch=angles_tot * SLICES) / SLICES / chunk_count
    debug_print(279, "rfft_plan", rfft_plan_slice_size, True)
    rfft_result_size = padded_tmp_p_input_slice / chunk_count
    debug_print(279, "rfft_result", rfft_result_size, True)
    filtered_rfft_result_size = rfft_result_size / chunk_count
    debug_print(279, "filtered_rfft_result", filtered_rfft_result_size, True)
    irfft_scratch_memory_size = filtered_rfft_result_size / chunk_count
    debug_print(280, "irfft_scratch", irfft_scratch_memory_size, True)

    datac_size = angles_tot * n * np.complex64().itemsize / 2
    debug_print(290, "datac_size", datac_size, True)
    fde_size = (2 * m + 2 * n) * (2 * m + 2 * n) * np.complex64().itemsize / 2
    debug_print(293, "fde_size", fde_size, True)
    fft_plan_slice_size = cufft_estimate_1d(nx=n,fft_type=CufftType.CUFFT_C2C,batch=angles_tot * SLICES) / SLICES
    debug_print(309, "fft_plan", fft_plan_slice_size, True)

    sorted_theta_indices_size = angles_tot * np.int64().itemsize
    debug_print(345, "sorted_indices", sorted_theta_indices_size, False)
    sorted_theta_size = angles_tot * np.float32().itemsize
    debug_print(346, "sorted_theta", sorted_theta_size, False)
    angle_range_size = center_size * center_size * (1 + angle_range_pi_count * 2) * np.int16().itemsize
    debug_print(351, "angle_range", angle_range_size, False)

    recon_output_size = n * n * np.float32().itemsize
    debug_print(434, "recon_up", recon_output_size, True)
    ifft2_plan_slice_size = cufft_estimate_2d(nx=(2 * m + 2 * n),ny=(2 * m + 2 * n),fft_type=CufftType.CUFFT_C2C) / 2
    debug_print(449, "ifft2_plan", ifft2_plan_slice_size, True)
    circular_mask_size = np.prod(output_dims) / 2 * np.int64().itemsize * 4
    debug_print(485, "circular_mask", circular_mask_size, False)
    after_recon_swapaxis_slice = np.prod(non_slice_dims_shape) * np.float32().itemsize
    debug_print(-1, "swapaxis", after_recon_swapaxis_slice, True)


    tot_memory_bytes = 0
    current_tot_memory_bytes = 0

    fixed_amount = 0
    current_fixed_amount = 0
    
    def bump_memory(ln, amount, per_slice: bool):
        nonlocal tot_memory_bytes
        nonlocal current_tot_memory_bytes
        nonlocal fixed_amount
        nonlocal current_fixed_amount

        if per_slice:
            current_tot_memory_bytes += amount
            tot_memory_bytes = max(tot_memory_bytes, current_tot_memory_bytes)
        else:
            current_fixed_amount += amount
            fixed_amount = max(fixed_amount, current_fixed_amount)
        
        debug_print(ln, "tot_memory_bytes", tot_memory_bytes, True)
        debug_print(ln, "current_tot_memory_bytes", current_tot_memory_bytes, True)
        debug_print(ln, "fixed_amount", fixed_amount, False)
        debug_print(ln, "current_fixed_amount", current_fixed_amount, False)
        debug_print(ln, "peak", tot_memory_bytes * ACTUAL_SLICE_COUNT + fixed_amount, False)
        debug_print(ln, "current_mem", current_tot_memory_bytes * ACTUAL_SLICE_COUNT + current_fixed_amount, False)
        print("****************")


    # bump_memory(-1, in_slice_size, True)
    bump_memory(233, padded_in_slice_size, True)

    bump_memory(240, theta_size, False)
    bump_memory(345, sorted_theta_indices_size, False)
    bump_memory(346, sorted_theta_size, False)
    bump_memory(351, angle_range_size, False)
    bump_memory(259, filter_size, False)
    bump_memory(262, rfftfreq_size, False)
    bump_memory(263, scaled_filter_size, False)

    bump_memory(266, tmp_p_input_slice, True)
    bump_memory(273, padded_tmp_p_input_slice, True)
    bump_memory(273, padded_tmp_p_input_slice, True)
    bump_memory(279, rfft_plan_slice_size, True)
    bump_memory(279, rfft_result_size, True)
    bump_memory(279, filtered_rfft_result_size, True)
    bump_memory(279, -rfft_result_size, True)
    bump_memory(279, -filtered_rfft_result_size, True)
    bump_memory(280, irfft_scratch_memory_size, True)
    bump_memory(283, -padded_tmp_p_input_slice, True)
    bump_memory(286, -padded_in_slice_size, True)

    bump_memory(286, -filter_size, False)
    bump_memory(286, -rfftfreq_size, False)
    bump_memory(286, -scaled_filter_size, False)

    bump_memory(290, datac_size, True)
    bump_memory(293, fde_size, True)
    bump_memory(307, -tmp_p_input_slice, True)
    bump_memory(309, fft_plan_slice_size, True)
    bump_memory(309, datac_size, True)

    bump_memory(424, -datac_size, True)
    bump_memory(434, recon_output_size, True)
    bump_memory(449, ifft2_plan_slice_size, True)
    bump_memory(483, -fde_size, True)

    bump_memory(485, circular_mask_size, False)
    bump_memory(-1, after_recon_swapaxis_slice, True)

    print(f"tot_memory_bytes: {tot_memory_bytes / 1024 / 1024} MB, fixed_amount: {fixed_amount / 1024 / 1024} MB")
    return (tot_memory_bytes*1.25, fixed_amount)
    # return (tot_memory_bytes * 1.25, fixed_amount)



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
