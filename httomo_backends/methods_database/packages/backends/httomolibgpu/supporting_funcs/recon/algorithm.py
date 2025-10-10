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


def _calc_memory_bytes_LPRec3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    # Based on: https://github.com/dkazanc/ToMoBAR/pull/112/commits/4704ecdc6ded3dd5ec0583c2008aa104f30a8a39

    detector_pad = 0
    if "detector_pad" in kwargs:
        detector_pad = kwargs["detector_pad"]
    if detector_pad is True:
        detector_pad = __estimate_detectorHoriz_padding(non_slice_dims_shape[1])
    elif detector_pad is False:
        detector_pad = 0        

    min_mem_usage_filter = False
    if "min_mem_usage_filter" in kwargs:
        min_mem_usage_filter = kwargs["min_mem_usage_filter"]
    min_mem_usage_ifft2 = False
    if "min_mem_usage_ifft2" in kwargs:
        min_mem_usage_ifft2 = kwargs["min_mem_usage_ifft2"]

    angles_tot = non_slice_dims_shape[0]
    DetectorsLengthH_prepad = non_slice_dims_shape[1]
    DetectorsLengthH = non_slice_dims_shape[1] + 2 * detector_pad
    SLICES = 200  # dummy multiplier+divisor to pass large batch size threshold
    _CENTER_SIZE_MIN = 192  # must be divisible by 8

    n = DetectorsLengthH

    odd_horiz = False
    if (n % 2) != 0:
        n = n + 1  # dealing with the odd horizontal detector size
        odd_horiz = True

    eps = 1e-4  # accuracy of usfft
    mu = -np.log(eps) / (2 * n * n)
    m = int(
        np.ceil(
            2 * n * 1 / np.pi * np.sqrt(-mu * np.log(eps) + (mu * n) * (mu * n) / 4)
        )
    )

    center_size = 32768
    center_size = min(center_size, n * 2)

    chunk_count = 4
    projection_chunk_count = 4
    oversampling_level = 4  # at least 3 or larger required
    power_of_2_oversampling = True

    if power_of_2_oversampling:
        ne = 2 ** math.ceil(math.log2(DetectorsLengthH_prepad * 3))
        if n > ne:
            ne = 2 ** math.ceil(math.log2(n))
    else:
        ne = int(oversampling_level * DetectorsLengthH_prepad)
        ne = max(ne, n)

    padding_m = ne // 2 - n // 2

    if "angles" in kwargs:
        angles = kwargs["angles"]
        sorted_theta_cpu = np.sort(angles)
        theta_full_range = abs(sorted_theta_cpu[angles_tot - 1] - sorted_theta_cpu[0])
        angle_range_pi_count = 1 + int(np.ceil(theta_full_range / math.pi))
        angle_range_pi_count += 1  # account for difference from actual algorithm
    else:
        angle_range_pi_count = 1 + int(
            np.ceil(2)
        )  # assume a 2 * PI projection angle range

    output_dims = __calc_output_dim_recon(non_slice_dims_shape, **kwargs)
    if odd_horiz:
        output_dims = tuple(x + 1 for x in output_dims)

    in_slice_size = (angles_tot * DetectorsLengthH) * dtype.itemsize
    padded_in_slice_size = angles_tot * n * np.float32().itemsize

    theta_size = angles_tot * np.float32().itemsize
    filter_size = (n // 2 + 1) * np.float32().itemsize
    rfftfreq_size = filter_size
    scaled_filter_size = filter_size

    tmp_p_input_slice = angles_tot * n * np.float32().itemsize

    padded_tmp_p_input_slice = angles_tot * (n + padding_m * 2) * np.float32().itemsize
    rfft_plan_slice_size = (
        cufft_estimate_1d(
            nx=n + padding_m * 2,
            fft_type=CufftType.CUFFT_R2C,
            batch=angles_tot * SLICES,
        )
        / SLICES
    )
    rfft_result_size = angles_tot * (n + padding_m * 2) * np.complex64().itemsize
    filtered_rfft_result_size = rfft_result_size
    irfft_plan_slice_size = (
        cufft_estimate_1d(
            nx=(n + padding_m * 2),
            fft_type=CufftType.CUFFT_C2R,
            batch=angles_tot * SLICES,
        )
        / SLICES
    )
    irfft_scratch_memory_size = filtered_rfft_result_size * 2
    irfft_result_size = angles_tot * (n + padding_m * 2) * np.float32().itemsize

    datac_size = angles_tot * n * np.complex64().itemsize / 2
    fde_size = 2 * n * 2 * n * np.complex64().itemsize / 2
    fft_plan_slice_size = (
        cufft_estimate_1d(nx=n, fft_type=CufftType.CUFFT_C2C, batch=angles_tot * SLICES)
        / SLICES
    )
    fft_result_size = datac_size

    sorted_theta_indices_size = angles_tot * np.int64().itemsize
    sorted_theta_size = angles_tot * np.float32().itemsize
    angle_range_size = (
        center_size * center_size * (1 + angle_range_pi_count * 2) * np.int16().itemsize
    )

    recon_output_size = (
        DetectorsLengthH_prepad * DetectorsLengthH_prepad * np.float32().itemsize
    )
    ifft2_plan_slice_size = (
        cufft_estimate_2d(
            nx=2 * n, ny=2 * n, fft_type=CufftType.CUFFT_C2C
        )
        / 2
    )
    circular_mask_size = np.prod(output_dims) / 2 * np.int64().itemsize * 4
    after_recon_swapaxis_slice = recon_output_size

    tot_memory_bytes = 0
    current_tot_memory_bytes = 0

    fixed_amount = 0
    current_fixed_amount = 0

    def add_to_memory_counters(amount, per_slice: bool):
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

    add_to_memory_counters(in_slice_size, True)
    add_to_memory_counters(padded_in_slice_size, True)

    add_to_memory_counters(theta_size, False)
    if center_size >= _CENTER_SIZE_MIN:
        add_to_memory_counters(sorted_theta_indices_size, False)
        add_to_memory_counters(sorted_theta_size, False)
        add_to_memory_counters(angle_range_size, False)
    add_to_memory_counters(filter_size, False)
    add_to_memory_counters(rfftfreq_size, False)
    add_to_memory_counters(scaled_filter_size, False)

    add_to_memory_counters(tmp_p_input_slice, True)
    if min_mem_usage_filter:
        add_to_memory_counters(rfft_plan_slice_size / 4 / projection_chunk_count, False)
        add_to_memory_counters(irfft_plan_slice_size / 4 / projection_chunk_count, False)
        add_to_memory_counters(padded_tmp_p_input_slice / projection_chunk_count, False)

        add_to_memory_counters(rfft_result_size / projection_chunk_count, False)
        add_to_memory_counters(filtered_rfft_result_size / projection_chunk_count, False)
        add_to_memory_counters(-rfft_result_size / projection_chunk_count, False)
        add_to_memory_counters(-padded_tmp_p_input_slice / projection_chunk_count, False)

        add_to_memory_counters(irfft_scratch_memory_size / projection_chunk_count, False)
        add_to_memory_counters(-irfft_scratch_memory_size / projection_chunk_count, False)
        add_to_memory_counters(irfft_result_size / projection_chunk_count, False)
        add_to_memory_counters(-filtered_rfft_result_size / projection_chunk_count, False)

        add_to_memory_counters(-irfft_result_size / projection_chunk_count, False)
    else:
        add_to_memory_counters(rfft_plan_slice_size / chunk_count / projection_chunk_count * 2, True)
        add_to_memory_counters(irfft_plan_slice_size / chunk_count / projection_chunk_count * 2, True)
        # add_to_memory_counters(irfft_scratch_memory_size / chunk_count / projection_chunk_count, True)
        for _ in range(0, chunk_count):
            add_to_memory_counters(padded_tmp_p_input_slice / chunk_count / projection_chunk_count, True)

            add_to_memory_counters(rfft_result_size / chunk_count / projection_chunk_count, True)
            add_to_memory_counters(filtered_rfft_result_size / chunk_count / projection_chunk_count, True)
            add_to_memory_counters(-rfft_result_size / chunk_count / projection_chunk_count, True)
            add_to_memory_counters(-padded_tmp_p_input_slice / chunk_count / projection_chunk_count, True)

            add_to_memory_counters(irfft_scratch_memory_size / chunk_count / projection_chunk_count, True)
            add_to_memory_counters(-irfft_scratch_memory_size / chunk_count / projection_chunk_count, True)
            add_to_memory_counters(irfft_result_size / chunk_count / projection_chunk_count, True)
            add_to_memory_counters(-filtered_rfft_result_size / chunk_count / projection_chunk_count, True)

            add_to_memory_counters(-irfft_result_size / chunk_count / projection_chunk_count, True)

    add_to_memory_counters(-padded_in_slice_size, True)
    add_to_memory_counters(-filter_size, False)
    add_to_memory_counters(-rfftfreq_size, False)
    add_to_memory_counters(-scaled_filter_size, False)

    add_to_memory_counters(datac_size, True)
    add_to_memory_counters(fde_size, True)
    add_to_memory_counters(-tmp_p_input_slice, True)
    add_to_memory_counters(fft_plan_slice_size, True)
    add_to_memory_counters(fft_result_size, True)
    add_to_memory_counters(-datac_size, True)

    add_to_memory_counters(-fft_result_size, True)

    if min_mem_usage_ifft2:
        add_to_memory_counters(ifft2_plan_slice_size, False)
        add_to_memory_counters(fde_size * 2, False)
        add_to_memory_counters(-fde_size * 2, False)
    else:
        add_to_memory_counters(ifft2_plan_slice_size / chunk_count * 2, True)
        for _ in range(0, chunk_count):
            add_to_memory_counters(fde_size / chunk_count, True)
            add_to_memory_counters(-fde_size / chunk_count, True)

    add_to_memory_counters(recon_output_size, True)
    add_to_memory_counters(-fde_size, True)
    add_to_memory_counters(circular_mask_size, False)
    add_to_memory_counters(after_recon_swapaxis_slice, True)

    if min_mem_usage_filter and min_mem_usage_ifft2:
        return (tot_memory_bytes * 1.1, fixed_amount)
    elif min_mem_usage_filter and not min_mem_usage_ifft2:
        return (tot_memory_bytes * 1.15, fixed_amount)
    else:
        return (tot_memory_bytes * 1.1, fixed_amount)


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

    astra_projection = (in_data_size + out_data_size)

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
    return int(np.sqrt(2 * (det_half**2)) // 2)
