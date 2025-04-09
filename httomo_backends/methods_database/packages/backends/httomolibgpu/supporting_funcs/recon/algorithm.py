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
    # Based on: https://github.com/neon60/ToMoBAR/commit/fe95670d2b02edb4036be0e10f38967976af94d5

    angles_tot = non_slice_dims_shape[0]
    DetectorsLengthH = non_slice_dims_shape[1]
    SLICES = 200  # dummy multiplier+divisor to pass large batch size threshold

    n = DetectorsLengthH

    odd_horiz = False
    if (n % 2) != 0:
        n = n - 1  # dealing with the odd horizontal detector size
        odd_horiz = True

    def debug_print(line_number: int, var_name: str, size_in_bytes: int) -> str:
        print(f"{line_number} - {var_name}: {size_in_bytes} B / {size_in_bytes / 1024} KB / {size_in_bytes / 1024 ** 2} MB / {size_in_bytes * 16 / 1024 ** 2} MB")

    eps = 1e-4  # accuracy of usfft
    mu = -np.log(eps) / (2 * n * n)
    m = int(
        np.ceil(
            2 * n * 1 / np.pi * np.sqrt(-mu * np.log(eps) + (mu * n) * (mu * n) / 4)
        )
    )

    center_size = 6144
    center_size = min(center_size, n * 2 + m * 2)
    print(f"m: {m}")
    print(f"center_size: {center_size}")

    oversampling_level = 2  # at least 2 or larger required
    ne = oversampling_level * n
    padding_m = ne // 2 - n // 2
    print(f"padding_m: {padding_m}")

    output_dims = __calc_output_dim_recon(non_slice_dims_shape, **kwargs)
    if odd_horiz:
        output_dims = tuple(x + 1 for x in output_dims)
    print(f"output_dims: {output_dims}")

    in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    debug_print(0, "in_slice_size", in_slice_size)
    padded_in_slice_size = np.prod(non_slice_dims_shape) * np.float32().itemsize
    debug_print(246, "padded_in_slice_size", padded_in_slice_size)
    theta_size = angles_tot * np.float32().itemsize
    debug_print(262, "theta_size", theta_size)
    recon_output_size = (n + 1) * (n + 1) * np.float32().itemsize if odd_horiz else n * n * np.float32().itemsize    # 264
    debug_print(281, "recon_output_size", recon_output_size)
    linspace_size = n * np.float32().itemsize
    debug_print(285, "linspace_size", linspace_size)
    meshgrid_size = 2 * n * n * np.float32().itemsize
    debug_print(286, "meshgrid_size", meshgrid_size)
    phi_size = 6 * n * n * np.float32().itemsize
    debug_print(287, "phi_size", phi_size)
    angle_range_size = center_size * center_size * 3 * np.int32().itemsize
    debug_print(293, "angle_range_size", angle_range_size)
    c1dfftshift_size = n * np.int8().itemsize
    debug_print(296, "c1dfftshift_size", c1dfftshift_size)
    c2dfftshift_slice_size = 4 * n * n * np.int8().itemsize
    debug_print(299, "c2dfftshift_slice_size", c2dfftshift_slice_size)
    filter_size = (n // 2 + 1) * np.float32().itemsize
    debug_print(309, "filter_size", filter_size)
    rfftfreq_size = filter_size
    debug_print(312, "rfftfreq_size", rfftfreq_size)
    scaled_filter_size = filter_size
    debug_print(313, "scaled_filter_size", scaled_filter_size)
    tmp_p_input_slice = np.prod(non_slice_dims_shape) * np.float32().itemsize
    debug_print(316, "tmp_p_input_slice", tmp_p_input_slice)
    padded_tmp_p_input_slice = angles_tot * (n + padding_m * 2) * dtype.itemsize
    debug_print(326, "padded_tmp_p_input_slice", padded_tmp_p_input_slice)
    rfft_result_size = padded_tmp_p_input_slice
    debug_print(327, "rfft_result_size", rfft_result_size)
    filtered_rfft_result_size = rfft_result_size
    debug_print(327, "filtered_rfft_result_size", filtered_rfft_result_size)
    rfft_plan_slice_size = cufft_estimate_1d(nx=(n + padding_m * 2),fft_type=CufftType.CUFFT_R2C,batch=angles_tot * SLICES) / SLICES
    debug_print(327, "rfft_plan_slice_size", rfft_plan_slice_size)
    irfft_result_size = filtered_rfft_result_size * 2
    debug_print(327, "irfft_result_size", irfft_result_size)
    irfft_plan_slice_size = cufft_estimate_1d(nx=(n + padding_m * 2),fft_type=CufftType.CUFFT_C2R,batch=angles_tot * SLICES) / SLICES
    debug_print(327, "irfft_plan_slice_size", irfft_plan_slice_size)
    conversion_to_complex_size = np.prod(non_slice_dims_shape) * np.complex64().itemsize / 2
    debug_print(333, "conversion_to_complex_size", conversion_to_complex_size)
    datac_size = np.prod(non_slice_dims_shape) * np.complex64().itemsize / 2
    debug_print(333, "datac_size", datac_size)
    fde_size = (2 * m + 2 * n) * (2 * m + 2 * n) * np.complex64().itemsize / 2
    debug_print(341, "fde_size", fde_size)
    shifted_datac_size = datac_size
    debug_print(344, "shifted_datac_size", shifted_datac_size)
    fft_result_size = datac_size
    debug_print(344, "fft_result_size", fft_result_size)
    backshifted_datac_size = datac_size
    debug_print(344, "backshifted_datac_size", backshifted_datac_size)
    scaled_backshifted_datac_size = datac_size
    debug_print(344, "scaled_backshifted_datac_size", scaled_backshifted_datac_size)
    fft_plan_slice_size = cufft_estimate_1d(nx=n,fft_type=CufftType.CUFFT_C2C,batch=angles_tot * SLICES) / SLICES
    debug_print(344, "fft_plan_slice_size", fft_plan_slice_size)
    fde_view_size = 4 * n * n * np.complex64().itemsize / 2
    shifted_fde_view_size = fde_view_size
    debug_print(474, "shifted_fde_view_size", shifted_fde_view_size)
    ifft2_plan_slice_size = cufft_estimate_2d(nx=(2 * n),ny=(2 * n),fft_type=CufftType.CUFFT_C2C) / 2
    debug_print(474, "ifft2_plan_slice_size", ifft2_plan_slice_size)
    fde2_size = n * n * np.complex64().itemsize / 2
    debug_print(479, "fde2_size", fde2_size)
    concatenate_size = fde2_size
    debug_print(485, "concatenate_size", concatenate_size)
    circular_mask_size = np.prod(output_dims) / 2 * np.int64().itemsize * 4
    debug_print(496, "circular_mask_size", circular_mask_size)

    after_recon_swapaxis_slice = np.prod(non_slice_dims_shape) * np.float32().itemsize
    debug_print(0, "after_recon_swapaxis_slice", after_recon_swapaxis_slice)

    scope_sums = [
        in_slice_size + padded_in_slice_size
        , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + tmp_p_input_slice + padded_tmp_p_input_slice + rfft_result_size + filtered_rfft_result_size + irfft_result_size
        , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + tmp_p_input_slice + datac_size + conversion_to_complex_size
        , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + datac_size + fde_size + fft_plan_slice_size + shifted_datac_size + fft_result_size + backshifted_datac_size + scaled_backshifted_datac_size
        , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fde_size + fft_plan_slice_size
        , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fft_plan_slice_size + ifft2_plan_slice_size + shifted_fde_view_size
        , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fft_plan_slice_size + ifft2_plan_slice_size + fde2_size + concatenate_size
        , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fft_plan_slice_size + ifft2_plan_slice_size + after_recon_swapaxis_slice
    ]

    tot_memory_bytes_peak = max(scope_sums)
    tot_memory_bytes = int(tot_memory_bytes_peak)

    fixed_amount = int(
        max(
            theta_size + phi_size + linspace_size + meshgrid_size
            , theta_size + phi_size + angle_range_size + c1dfftshift_size + c2dfftshift_slice_size + filter_size + rfftfreq_size + scaled_filter_size
            , theta_size + phi_size + circular_mask_size
        )
    )

    print(f"tot_memory_bytes: {tot_memory_bytes}")
    print(f"fixed_amount: {fixed_amount}")

    return (tot_memory_bytes, fixed_amount)



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
