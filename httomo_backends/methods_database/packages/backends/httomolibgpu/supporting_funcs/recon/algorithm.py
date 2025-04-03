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
    angles_tot = non_slice_dims_shape[0]
    DetectorsLengthH = non_slice_dims_shape[1]
    SLICES = 200  # dummy multiplier+divisor to pass large batch size threshold

    det_height = angles_tot
    n = DetectorsLengthH

    odd_horiz = False
    if (n % 2) != 0:
        n = n - 1  # dealing with the odd horizontal detector size
        odd_horiz = True

    eps = 1e-4  # accuracy of usfft
    mu = -np.log(eps) / (2 * n * n)
    m = int(
        np.ceil(
            2
            * n
            * 1
            / np.pi
            * np.sqrt(
                -mu * np.log(eps)
                + (mu * n) * (mu * n) / 4
            )
        )
    )

    center_size = 6144
    center_size = min(center_size, n * 2 + m * 2)

    padding_m = n - n // 2

    output_dims = __calc_output_dim_recon(non_slice_dims_shape, **kwargs)

    in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    padded_in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize   # 232 deleted
    theta_size = angles_tot * np.float32().itemsize # 245
    recon_output_size = np.prod(output_dims) * np.float32().itemsize    # 264
    linspace_size = n * np.float32().itemsize # 268 deleted
    meshgrid_size = n * n * np.float32().itemsize # 269 deleted
    phi_size = n * n * np.float32().itemsize # 270
    angle_range_size = center_size * center_size * 3 * np.int32().itemsize # 276 deleted
    c1dfftshift_size = n * np.int8().itemsize   # 279 deleted
    c2dfftshift_slice_size = (np.prod(4 * n * n) * np.int8().itemsize) # 282 deleted
    filter_size = (n // 2 + 1) * np.float32().itemsize # 292 deleted
    wfilter_size = filter_size # 295
    scaled_filter_size = filter_size # 296
    tmp_p_input_slice = np.prod(non_slice_dims_shape) * dtype.itemsize # 299
    # 307 negligible
    # 308 negligible
    padded_tmp_p_input_slice = tmp_p_input_slice # 321
    rfft_plan_slice_size = cufft_estimate_1d(nx=n,fft_type=CufftType.CUFFT_R2C,batch=det_height * SLICES) / SLICES # 322
    irfft_plan_slice_size = cufft_estimate_1d(nx=n,fft_type=CufftType.CUFFT_C2R,batch=det_height * SLICES) / SLICES # 322
    data_c_size = np.prod(0.5 * angles_tot * n) * np.complex64().itemsize # 329 deleted
    fde_size = ((angles_tot // 2) * (2 * m + 2 * n) * (2 * m + 2 * n)) * np.complex64().itemsize # 336 deleted
    fft_plan_slice_size = cufft_estimate_1d(nx=n,fft_type=CufftType.CUFFT_C2C,batch=det_height * SLICES) / SLICES # 339
    # 364 negligible
    # 377 negligible
    # 408 negligible
    # 409 negligible
    ifft2_plan_slice_size = cufft_estimate_2d(nx=DetectorsLengthH,ny=angles_tot,fft_type=CufftType.CUFFT_C2C) #  468
    fde2_size = fde_size # 473
    recon_output_size_again = 2 * fde2_size # 479
    circular_mask_size = np.prod(output_dims) * np.int64().itemsize # 490

    tot_memory_bytes = int(
        max(
            in_slice_size + padded_in_slice_size
            , in_slice_size + recon_output_size + linspace_size + meshgrid_size
            , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + tmp_p_input_slice + padded_tmp_p_input_slice
            , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + tmp_p_input_slice + data_c_size
            , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + data_c_size + fde_size + fft_plan_slice_size
            , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fde_size + fft_plan_slice_size + fde2_size
            , in_slice_size + recon_output_size + rfft_plan_slice_size + irfft_plan_slice_size + fft_plan_slice_size + fde2_size + ifft2_plan_slice_size + recon_output_size_again
        )
    )

    fixed_amount = int(
        max(
            theta_size + phi_size + angle_range_size + c1dfftshift_size + c2dfftshift_slice_size + filter_size + wfilter_size + scaled_filter_size
            , theta_size + phi_size + circular_mask_size
        )
    )

    return (1.2 * tot_memory_bytes, 1.2 * fixed_amount)



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
