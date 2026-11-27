"""CPU checks for the PD-TV scheduling estimate."""

import numpy as np

from httomo_backends.methods_database.query import MethodsDatabaseQuery


def query():
    return MethodsDatabaseQuery("httomolibgpu.misc.denoise", "total_variation_PD")


def test_pd_tv_uses_module_estimator():
    memory = query().get_memory_gpu_params()
    assert memory.method == "module"
    assert memory.multiplier is None


def test_pd_tv_half_precision_reduces_workspace():
    full, fixed = query().calculate_memory_bytes((100, 200), np.dtype("float32"))
    half, half_fixed = query().calculate_memory_bytes(
        (100, 200), np.dtype("float32"), half_precision=True
    )
    assert half < full
    # Primal image buffers remain float32; memory does not halve.
    assert half > full / 2
    assert fixed == half_fixed


def test_pd_tv_memory_scales_with_slice_area():
    small, fixed = query().calculate_memory_bytes((100, 200), np.dtype("float32"))
    large, large_fixed = query().calculate_memory_bytes((200, 200), np.dtype("float32"))
    assert large == 2 * small
    assert fixed == large_fixed


def test_pd_tv_default_matches_explicit_full_precision():
    default = query().calculate_memory_bytes((100, 200), np.dtype("float32"))
    explicit = query().calculate_memory_bytes(
        (100, 200), np.dtype("float32"), half_precision=False
    )
    assert default == explicit
