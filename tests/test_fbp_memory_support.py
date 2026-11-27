"""Sizing checks for FBP3d's module estimator, without a CUDA device."""

import numpy as np
import pytest

from httomo_backends.methods_database.packages.backends.httomolibgpu.supporting_funcs.recon import (
    algorithm,
)


@pytest.mark.parametrize(
    "width,padding,expected",
    [
        (64, False, 128),
        (65, False, 256),
        (96, False, 256),
        (128, False, 256),
        (129, False, 512),
        (64, 1, 256),
        (65, 7, 256),
        (96, True, 512),
    ],
)
def test_fbp_fft_workspaces_use_zero_padded_length(mocker, width, padding, expected):
    estimate = mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, width),
        np.dtype("float32"),
        detector_pad=padding,
    )
    assert estimate.call_count == 2
    assert [call.kwargs["nx"] for call in estimate.call_args_list] == [
        expected,
        expected,
    ]
    assert [call.kwargs["fft_type"] for call in estimate.call_args_list] == [
        algorithm.CufftType.CUFFT_R2C,
        algorithm.CufftType.CUFFT_C2R,
    ]


def test_fbp_spectrum_and_response_grow_at_fft_boundary(mocker):
    mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    _, fixed64 = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 64),
        np.dtype("float32"),
        recon_size=1,
    )
    _, fixed65 = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 65),
        np.dtype("float32"),
        recon_size=1,
    )
    assert fixed64 >= 65 * np.dtype("float32").itemsize
    assert fixed65 - fixed64 == 64 * np.dtype("float32").itemsize


def test_fbp_retains_both_fft_workspaces(mocker):
    estimate = mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    baseline, _ = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 65), np.dtype("float32")
    )
    # The estimator normalises a representative 200-slice cuFFT batch.
    estimate.side_effect = [200 * 1024, 200 * 2048]
    with_plans, _ = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 65), np.dtype("float32")
    )
    assert with_plans - baseline == 1024 + 2048


def test_fbp_fft_expansion_exceeds_detector_only_growth(mocker):
    mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    small, _ = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 64), np.dtype("float32"), recon_size=32
    )
    large, _ = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 65), np.dtype("float32"), recon_size=32
    )
    # One extra detector column crosses an FFT boundary; workspace must grow
    # far more than the one-column spatial buffers.
    assert large - small > 60 * 128 * np.dtype("float32").itemsize


@pytest.mark.parametrize("padding", [False, True, 7])
@pytest.mark.parametrize("defaults", [{}, {"recon_size": None}])
def test_fbp_padding_preserves_default_reconstruction_size(mocker, padding, defaults):
    mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    shape = (60, 96)
    default = algorithm._calc_memory_bytes_FBP3d_tomobar(
        shape,
        np.dtype("float32"),
        detector_pad=padding,
        **defaults,
    )
    explicit = algorithm._calc_memory_bytes_FBP3d_tomobar(
        shape,
        np.dtype("float32"),
        detector_pad=padding,
        recon_size=96,
    )
    assert default == explicit


@pytest.mark.parametrize("padding", [False, True, 7])
def test_fbp_explicit_reconstruction_size_still_controls_output_memory(mocker, padding):
    mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    small, small_fixed = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 96),
        np.dtype("float32"),
        detector_pad=padding,
        recon_size=32,
    )
    large, large_fixed = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 96),
        np.dtype("float32"),
        detector_pad=padding,
        recon_size=128,
    )
    assert large >= small
    assert large_fixed > small_fixed
    assert 3 * large + large_fixed > 3 * small + small_fixed


def test_fbp_mask_can_be_disabled(mocker):
    mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    enabled = algorithm._calc_memory_bytes_FBP3d_tomobar((60, 96), np.dtype("float32"))
    disabled = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 96),
        np.dtype("float32"),
        recon_mask_radius=None,
    )
    assert enabled[0] == disabled[0]
    assert enabled[1] > disabled[1]


def test_fbp_mask_reserves_two_distance_planes_and_coordinates(mocker):
    mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    _, fixed = algorithm._calc_memory_bytes_FBP3d_tomobar((60, 96), np.dtype("float32"))
    assert (
        fixed
        >= 2 * 96 * 96 * np.dtype("float64").itemsize
        + 2 * 96 * np.dtype("int64").itemsize
    )


def test_fbp_uint16_retains_original_input_and_float32_working_storage(mocker):
    mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    full, _ = algorithm._calc_memory_bytes_FBP3d_tomobar((60, 96), np.dtype("float32"))
    integer, _ = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 96), np.dtype("uint16")
    )
    # Working arrays stay float32; only the retained original input is smaller.
    assert full - integer == 60 * 96 * (4 - 2)


def test_fbp_padding_does_not_pad_the_retained_original_input(mocker):
    mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    full, _ = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 96),
        np.dtype("float32"),
        detector_pad=7,
    )
    integer, _ = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (60, 96),
        np.dtype("uint16"),
        detector_pad=7,
    )
    assert full - integer == 60 * 96 * (4 - 2)


def test_fbp_output_layout_copy_is_reserved(mocker):
    mocker.patch.object(algorithm, "cufft_estimate_1d", return_value=0)
    per_slice, _ = algorithm._calc_memory_bytes_FBP3d_tomobar(
        (2, 8),
        np.dtype("float32"),
        recon_size=256,
        recon_mask_radius=None,
    )
    assert per_slice >= 2 * 256 * 256 * np.dtype("float32").itemsize
