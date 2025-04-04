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
from scipy.interpolate import griddata

__all__ = [
    "_calc_memory_bytes_FBP",
    "_calc_memory_bytes_LPRec",
    "_calc_memory_bytes_SIRT",
    "_calc_memory_bytes_CGLS",
    "_calc_output_dim_FBP",
    "_calc_output_dim_LPRec",
    "_calc_output_dim_SIRT",
    "_calc_output_dim_CGLS",
]


def get_available_gpu_memory(safety_margin_percent: float = 10.0) -> int:
    try:
        import cupy as cp

        dev = cp.cuda.Device(0)  # we consider here only the first device
        with dev:
            pool = cp.get_default_memory_pool()
            available_memory = dev.mem_info[0] + pool.free_bytes()
            return int(available_memory * (1 - safety_margin_percent / 100.0))
    except:
        return int(100e9)  # arbitrarily high number - only used if GPU isn't available


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


def _calc_output_dim_LPRec(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_FBP(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_SIRT(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_CGLS(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_memory_bytes_FBP(
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
    output_dims = _calc_output_dim_FBP(non_slice_dims_shape, **kwargs)
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


def _calc_memory_bytes_LPRec(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    angles_tot = non_slice_dims_shape[0]
    DetectorsLengthH = non_slice_dims_shape[1]

    # calculate the output shape
    output_dims = __calc_output_dim_recon(non_slice_dims_shape, **kwargs)

    # input and and output slices
    in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_slice_size = np.prod(output_dims) * dtype.itemsize

    if DetectorsLengthH == 2560 and output_dims[0] == 2560:
        # The empirical calculation of slices based on the fixed value of DetectorsLengthH and reconstruction array sizes
        x = np.array([4, 8, 16, 32, 40])  # Memory in GB available on the device
        y = np.array(
            [10, 450, 900, 1800, 2700, 3600, 4500, 6000, 7200, 9000]
        )  # the number of projections in the data

        points = np.zeros((50, 2), dtype=int)
        index1 = 0
        index2 = 0
        for index_x in range(len(x)):
            for index_y in range(len(y)):
                points[index1] = [y[index_y], x[index2]]
                index1 += 1
            index2 += 1

        gb4_bites = 4e9
        gb8_bites = 8e9
        gb16_bites = 1.6e10
        gb32_bites = 3.2e10
        gb40_bites = 4e10

        # Here the total memory in bytes on a card divided by the number of slices that can fit that memory.
        # For instance for 8GB GPU card with 1800 x 2560 sinogram size given, you can fit 17 sinograms
        # before the OOM error occurs on the device.
        safety_margin = 1  # in the number of slices to reduce the memory usage
        memory_bytes_estimated_per_slice = np.array(
            [
                [
                    gb4_bites / (10 - safety_margin),
                    gb8_bites / (20 - safety_margin),
                    gb16_bites / (40 - safety_margin),
                    gb32_bites / (98 - safety_margin),
                    gb40_bites / (121 - safety_margin),
                ],
                [
                    gb4_bites / (10 - safety_margin),
                    gb8_bites / (20 - safety_margin),
                    gb16_bites / (40 - safety_margin),
                    gb32_bites / (90 - safety_margin),
                    gb40_bites / (107 - safety_margin),
                ],
                [
                    gb4_bites / (9 - safety_margin),
                    gb8_bites / (19 - safety_margin),
                    gb16_bites / (38 - safety_margin),
                    gb32_bites / (82 - safety_margin),
                    gb40_bites / (97 - safety_margin),
                ],
                [
                    gb4_bites / (8 - safety_margin),
                    gb8_bites / (17 - safety_margin),
                    gb16_bites / (34 - safety_margin),
                    gb32_bites / (70 - safety_margin),
                    gb40_bites / (82 - safety_margin),
                ],
                [
                    gb4_bites / (7 - safety_margin),
                    gb8_bites / (15 - safety_margin),
                    gb16_bites / (30 - safety_margin),
                    gb32_bites / (62 - safety_margin),
                    gb40_bites / (73 - safety_margin),
                ],
                [
                    gb4_bites / (6 - safety_margin),
                    gb8_bites / (13 - safety_margin),
                    gb16_bites / (26 - safety_margin),
                    gb32_bites / (56 - safety_margin),
                    gb40_bites / (63 - safety_margin),
                ],
                [
                    gb4_bites / (5 - safety_margin),
                    gb8_bites / (11 - safety_margin),
                    gb16_bites / (20 - safety_margin),
                    gb32_bites / (46 - safety_margin),
                    gb40_bites / (54 - safety_margin),
                ],
                [
                    gb4_bites / (4 - safety_margin),
                    gb8_bites / (8 - safety_margin),
                    gb16_bites / (16 - safety_margin),
                    gb32_bites / (32 - safety_margin),
                    gb40_bites / (40 - safety_margin),
                ],
                [
                    gb4_bites / (3 - safety_margin),
                    gb8_bites / (6 - safety_margin),
                    gb16_bites / (12 - safety_margin),
                    gb32_bites / (25 - safety_margin),
                    gb40_bites / (33 - safety_margin),
                ],
                [
                    gb4_bites / 1,
                    gb8_bites / (4 - safety_margin),
                    gb16_bites / (10 - safety_margin),
                    gb32_bites / (22 - safety_margin),
                    gb40_bites / (26 - safety_margin),
                ],
            ]
        )

        # now we can interpolate if the provided data lies within the range given
        available_memory = get_available_gpu_memory(
            10.0
        )  # NOTE: We check here ONLY the first device
        available_memory_in_GB = round(available_memory / (1024**3), 2)
        point_required = [angles_tot, available_memory_in_GB]
        # we interpolate in order to get the number of slices for a given angles and given memory amount
        result = griddata(
            points,
            memory_bytes_estimated_per_slice.T.flatten(),
            point_required,
            method="cubic",
        )
        # convert to bytes
        tot_memory_bytes = int(*result)
        fixed_amount = 0
    else:
        # mathematical calculation of the required memory
        n = DetectorsLengthH
        odd_horiz = False
        if (n % 2) != 0:
            n = n - 1  # dealing with the odd horizontal detector size
            odd_horiz = True

        eps = 1e-4  # accuracy of usfft
        mu = -np.log(eps) / (2 * n * n)
        m = int(
            np.ceil(
                2 * n * 1 / np.pi * np.sqrt(-mu * np.log(eps) + (mu * n) * (mu * n) / 4)
            )
        )

        data_c_size = np.prod(0.5 * angles_tot * n) * np.complex64().itemsize

        fde_size = (0.5 * (2 * m + 2 * n) * (2 * m + 2 * n)) * np.complex64().itemsize

        c1dfftshift_size = n * np.int8().itemsize

        c2dfftshift_slice_size = np.prod(4 * n * n) * np.int8().itemsize

        theta_size = angles_tot * np.float32().itemsize
        filter_size = (n // 2 + 1) * np.float32().itemsize
        freq_slice = angles_tot * (n + 1) * np.complex64().itemsize
        fftplan_size = freq_slice * 2

        phi_size = n * n * np.float32().itemsize

        # We have two fde arrays and sometimes need double fde.
        max_memory_per_slice = max(data_c_size + 2 * fde_size, 3 * fde_size)

        tot_memory_bytes = int(in_slice_size + out_slice_size + max_memory_per_slice)

        fixed_amount = int(
            fde_size
            + data_c_size
            + theta_size
            + fftplan_size
            + filter_size
            + phi_size
            + c1dfftshift_size
            + c2dfftshift_slice_size
            + freq_slice
        )

    return (tot_memory_bytes, fixed_amount)


def _calc_memory_bytes_SIRT(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    DetectorsLengthH = non_slice_dims_shape[1]
    # calculate the output shape
    output_dims = _calc_output_dim_SIRT(non_slice_dims_shape, **kwargs)

    in_data_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_data_size = np.prod(output_dims) * dtype.itemsize

    astra_projection = 2.5 * (in_data_size + out_data_size)

    tot_memory_bytes = 2 * in_data_size + 2 * out_data_size + astra_projection
    return (tot_memory_bytes, 0)


def _calc_memory_bytes_CGLS(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    DetectorsLengthH = non_slice_dims_shape[1]
    # calculate the output shape
    output_dims = _calc_output_dim_CGLS(non_slice_dims_shape, **kwargs)

    in_data_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_data_size = np.prod(output_dims) * dtype.itemsize

    astra_projection = 2.5 * (in_data_size + out_data_size)

    tot_memory_bytes = 2 * in_data_size + 2 * out_data_size + astra_projection
    return (tot_memory_bytes, 0)
