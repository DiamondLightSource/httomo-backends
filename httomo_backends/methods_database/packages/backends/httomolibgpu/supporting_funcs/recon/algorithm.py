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
from httomolibgpu.recon.algorithm import LPRec3d_tomobar
from tomobar.reconstruction.memory import DeviceMemStack
from tomobar._filters.ramlak import ramlak_fft_length

__all__ = [
    "_calc_memory_bytes_FBP3d_tomobar",
    "_calc_memory_bytes_for_slices_LPRec3d_tomobar",
    "_calc_memory_bytes_LPRec3d_tomobar",
    "_calc_memory_bytes_SIRT3d_tomobar",
    "_calc_memory_bytes_CGLS3d_tomobar",
    "_calc_memory_bytes_FISTA3d_tomobar",
    "_calc_memory_bytes_ADMM3d_tomobar",
    "_calc_memory_bytes_OSEM3d_tomobar",
    "_calc_output_dim_FBP2d_astra",
    "_calc_output_dim_FBP3d_tomobar",
    "_calc_output_dim_LPRec3d_tomobar",
    "_calc_output_dim_SIRT3d_tomobar",
    "_calc_output_dim_CGLS3d_tomobar",
    "_calc_output_dim_FISTA3d_tomobar",
    "_calc_output_dim_ADMM3d_tomobar",
    "_calc_output_dim_OSEM3d_tomobar",
    "_calc_padding_FISTA3d_tomobar",
    "_calc_padding_ADMM3d_tomobar",
    "_calc_padding_OSEM3d_tomobar",
]


def _calc_padding_FISTA3d_tomobar(**kwargs) -> Tuple[int, int]:
    return (5, 5)


def _calc_padding_ADMM3d_tomobar(**kwargs) -> Tuple[int, int]:
    return (5, 5)


def _calc_padding_OSEM3d_tomobar(**kwargs) -> Tuple[int, int]:
    return (5, 5)


def __calc_output_dim_recon(non_slice_dims_shape, **kwargs):
    """Function to calculate output dimensions for all reconstructors.
    The change of the dimension depends either on the user-provided "recon_size"
    parameter or taken as the size of the horizontal detector (default).

    """
    DetectorsLengthH = non_slice_dims_shape[1]
    recon_size = kwargs.get("recon_size", None)
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


def _calc_output_dim_ADMM3d_tomobar(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_OSEM3d_tomobar(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_memory_bytes_FBP3d_tomobar(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    """Estimate FBP's peak live buffers as per-slice bytes and fixed overhead.

    Reserve float32 packing even for C-contiguous input, since its layout is
    unknown here. FFT plans remain cached through reconstruction/output packing.
    ASTRA workspace retains a conservative one-projection/one-volume allowance;
    it is not measured by CuPy allocation hooks. cuFFT workspace is normalised
    from a representative 200-slice batch, as in the other FFT estimates.
    CUDA context, kernel compilation and unused pool blocks are not modelled.
    """
    angles, width = (int(size) for size in non_slice_dims_shape)
    detector_pad = kwargs.get("detector_pad", False)
    if detector_pad is True:
        detector_pad = __estimate_detectorHoriz_padding(width)
    elif detector_pad is False:
        detector_pad = 0
    padded_width = width + 2 * detector_pad
    fft_length = ramlak_fft_length(padded_width)
    float_bytes = np.dtype(np.float32).itemsize

    # Original input stays live in the wrapper throughout the call. Preparation
    # may convert/pack it and then edge-pad it in a separate float32 allocation.
    input_bytes = angles * width * dtype.itemsize
    packed_input = angles * width * float_bytes
    projection = angles * padded_width * float_bytes
    preparation = packed_input + (projection if detector_pad else 0)

    slices = 200
    fft_workspace = (
        sum(
            cufft_estimate_1d(nx=fft_length, fft_type=kind, batch=angles * slices)
            for kind in (CufftType.CUFFT_R2C, CufftType.CUFFT_C2R)
        )
        / slices
    )
    spectrum = angles * (fft_length // 2 + 1) * np.dtype(np.complex64).itemsize
    real_fft = angles * fft_length * float_bytes
    response = (fft_length // 2 + 1) * float_bytes

    # The prepared input, spectrum, padded inverse result and cropped result
    # coexist. This also bounds the forward FFT's zero-padded real input.
    filtering = projection + spectrum + real_fft + projection
    # Packing (angles, rows, columns) into (rows, angles, columns) happens
    # before the prepared input is released.
    projection_layout = 3 * projection

    output_dims = _calc_output_dim_FBP3d_tomobar(non_slice_dims_shape, **kwargs)
    output_pixels = math.prod(int(size) for size in output_dims)
    volume = output_pixels * float_bytes
    astra_workspace = projection + volume
    backprojection = projection + volume + astra_workspace
    masking = projection + volume
    # The wrapper packs the final (Y, detector rows, X) transpose while the
    # original reconstruction and cached FFT plans still exist.
    output_layout = 2 * volume

    mask_workspace = 0
    if kwargs.get("recon_mask_radius", 0.95) is not None:
        # sqrt holds an int64 distance plane and a float64 result together.
        # Reserve coordinate vectors/temporaries too; masking itself is in-place.
        mask_workspace = (
            2 * output_pixels + 3 * sum(int(size) for size in output_dims)
        ) * np.dtype(np.int64).itemsize

    per_slice = math.ceil(
        input_bytes
        + fft_workspace
        + max(
            preparation,
            filtering,
            projection_layout,
            backprojection,
            masking,
            output_layout,
        )
    )
    # Fixed masks and filter responses belong to separate stages. Combining
    # the largest per-slice and fixed terms is a conservative linear bound.
    fixed = max(response, mask_workspace) + 32 * 512
    return per_slice, int(fixed)


def _calc_memory_bytes_for_slices_LPRec3d_tomobar(
    dims_shape: Tuple[int, int, int],
    dtype: np.dtype,
    **kwargs,
) -> int:
    with DeviceMemStack():
        kwargs["data_dtype"] = dtype
        return LPRec3d_tomobar(dims_shape, **kwargs)


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
    power_of_2_cropping = False
    if "power_of_2_cropping" in kwargs:
        power_of_2_cropping = kwargs["power_of_2_cropping"]

    angles_tot = non_slice_dims_shape[0]
    DetectorsLengthH_prepad = non_slice_dims_shape[1]
    DetectorsLengthH = non_slice_dims_shape[1] + 2 * detector_pad
    SLICES = 200  # dummy multiplier+divisor to pass large batch size threshold
    _CENTER_SIZE_MIN = 192  # must be divisible by 8

    n = DetectorsLengthH
    if power_of_2_cropping:
        n_pow2 = 2 ** math.ceil(math.log2(n))
        if 0.9 < n / n_pow2:
            n = n_pow2

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
        cufft_estimate_2d(nx=2 * n, ny=2 * n, fft_type=CufftType.CUFFT_C2C) / 2
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
        add_to_memory_counters(
            irfft_plan_slice_size / 4 / projection_chunk_count, False
        )
        add_to_memory_counters(padded_tmp_p_input_slice / projection_chunk_count, False)

        add_to_memory_counters(rfft_result_size / projection_chunk_count, False)
        add_to_memory_counters(
            filtered_rfft_result_size / projection_chunk_count, False
        )
        add_to_memory_counters(-rfft_result_size / projection_chunk_count, False)
        add_to_memory_counters(
            -padded_tmp_p_input_slice / projection_chunk_count, False
        )

        add_to_memory_counters(
            irfft_scratch_memory_size / projection_chunk_count, False
        )
        add_to_memory_counters(
            -irfft_scratch_memory_size / projection_chunk_count, False
        )
        add_to_memory_counters(irfft_result_size / projection_chunk_count, False)
        add_to_memory_counters(
            -filtered_rfft_result_size / projection_chunk_count, False
        )

        add_to_memory_counters(-irfft_result_size / projection_chunk_count, False)
    else:
        add_to_memory_counters(
            rfft_plan_slice_size / chunk_count / projection_chunk_count * 2, True
        )
        add_to_memory_counters(
            irfft_plan_slice_size / chunk_count / projection_chunk_count * 2, True
        )
        # add_to_memory_counters(irfft_scratch_memory_size / chunk_count / projection_chunk_count, True)
        for _ in range(0, chunk_count):
            add_to_memory_counters(
                padded_tmp_p_input_slice / chunk_count / projection_chunk_count, True
            )

            add_to_memory_counters(
                rfft_result_size / chunk_count / projection_chunk_count, True
            )
            add_to_memory_counters(
                filtered_rfft_result_size / chunk_count / projection_chunk_count, True
            )
            add_to_memory_counters(
                -rfft_result_size / chunk_count / projection_chunk_count, True
            )
            add_to_memory_counters(
                -padded_tmp_p_input_slice / chunk_count / projection_chunk_count, True
            )

            add_to_memory_counters(
                irfft_scratch_memory_size / chunk_count / projection_chunk_count, True
            )
            add_to_memory_counters(
                -irfft_scratch_memory_size / chunk_count / projection_chunk_count, True
            )
            add_to_memory_counters(
                irfft_result_size / chunk_count / projection_chunk_count, True
            )
            add_to_memory_counters(
                -filtered_rfft_result_size / chunk_count / projection_chunk_count, True
            )

            add_to_memory_counters(
                -irfft_result_size / chunk_count / projection_chunk_count, True
            )

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
        return (tot_memory_bytes * 1.15, fixed_amount)
    elif min_mem_usage_filter and not min_mem_usage_ifft2:
        return (tot_memory_bytes + 60 * 1024 * 1024, fixed_amount)
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


def _calc_memory_bytes_OSEM3d_tomobar(
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
    output_dims = _calc_output_dim_OSEM3d_tomobar(non_slice_dims_shape, **kwargs)
    recon_data_size_original = (
        np.prod(output_dims) * dtype.itemsize
    )  # recon user-defined size

    in_data_siz_pad = (anglesnum * DetectorsLengthH_padded) * dtype.itemsize
    output_dims_larger_grid = (DetectorsLengthH_padded, DetectorsLengthH_padded)

    out_data_size = np.prod(output_dims_larger_grid) * dtype.itemsize
    normalisation = out_data_size
    Ax = in_data_siz_pad
    ratio = in_data_siz_pad
    backproj = out_data_size

    mlem_part = (
        recon_data_size_original + normalisation + Ax + ratio + backproj + out_data_size
    )
    regul_part = 8 * np.prod(output_dims_larger_grid) * dtype.itemsize

    tot_memory_bytes = int(mlem_part + regul_part)
    return (tot_memory_bytes, 0)


def _calc_memory_bytes_ADMM3d_tomobar(
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

    out_data_size = np.prod(output_dims_larger_grid) * dtype.itemsize
    x0 = out_data_size
    x = out_data_size
    z = out_data_size
    u = out_data_size
    grad_data = out_data_size
    grad_admm = out_data_size
    arithmetic = 3 * out_data_size

    admm_part = (
        recon_data_size_original
        + in_data_siz_pad
        + x0
        + x
        + z
        + u
        + grad_data
        + grad_admm
        + out_data_size
        + arithmetic
    )
    # Extra native full-data work for the optional CG/weighted objectives.
    if kwargs.get("data_solver", "linearised") == "cg":
        admm_part += 3 * out_data_size
    if kwargs.get("data_fidelity", "LS") in ("PWLS", "SWLS"):
        admm_part += in_data_siz_pad  # Owned, padded normalised weights.
    if kwargs.get("data_fidelity", "LS") == "SWLS":
        admm_part += 3 * DetectorsLengthH_padded * dtype.itemsize

    regul_part = 8 * np.prod(output_dims_larger_grid) * dtype.itemsize

    tot_memory_bytes = int(admm_part + regul_part)
    return (tot_memory_bytes, 0)


def __estimate_detectorHoriz_padding(detX_size) -> int:
    det_half = detX_size // 2
    padded_value_exact = int(np.sqrt(2 * (det_half**2))) - det_half
    padded_add_margin = padded_value_exact // 2
    return padded_value_exact + padded_add_margin
