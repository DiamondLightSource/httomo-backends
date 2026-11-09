"""NLM scheduling and GPU memory estimates, with optional CUDA measurements."""

import numpy as np
import pytest

from httomo_backends.methods_database.query import MethodsDatabaseQuery, Pattern
from httomo_backends.methods_database.packages.backends.httomolibgpu.supporting_funcs.misc.denoise import (
    _calc_memory_bytes_non_local_means,
    _calc_padding_non_local_means,
)


def test_nlm_database_registration():
    query = MethodsDatabaseQuery("httomolibgpu.misc.denoise", "non_local_means")
    # Axial slabs also make zero overlap valid for independent-slice mode.
    assert query.get_pattern() == Pattern.all
    assert query.get_implementation() == "gpu_cupy"
    assert query.get_output_dims_change() is False
    assert query.save_result_default() is False
    assert query.padding() is True
    assert query.get_memory_gpu_params().method == "module"
    assert query.calculate_padding() == (3, 3)
    assert query.calculate_memory_bytes((100, 200), np.dtype("float32")) == (
        264000,
        2816,
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, (3, 3)),
        ({"search_radius": 4, "patch_radius": 2}, (6, 6)),
        ({"search_radius": 0, "patch_radius": 0}, (0, 0)),
        ({"full_3d": False}, (0, 0)),
        ({"full_3d": False, "search_radius": 4, "patch_radius": 2}, (0, 0)),
    ],
)
def test_nlm_overlap(kwargs, expected):
    assert _calc_padding_non_local_means(**kwargs) == expected


def test_nlm_memory_scales_with_slice_area_but_weights_are_fixed():
    small, fixed = _calc_memory_bytes_non_local_means((100, 200), np.dtype("float32"))
    large, large_fixed = _calc_memory_bytes_non_local_means(
        (200, 200), np.dtype("float32")
    )
    assert large == 2 * small
    assert fixed == large_fixed
    assert small >= 3 * 100 * 200 * 4


def test_nlm_patch_radius_only_changes_fixed_workspace():
    baseline, fixed = _calc_memory_bytes_non_local_means(
        (100, 200), np.dtype("float32")
    )
    larger, larger_fixed = _calc_memory_bytes_non_local_means(
        (100, 200), np.dtype("float32"), patch_radius=100
    )
    assert baseline == larger
    assert larger_fixed > fixed


def test_nlm_selected_neighbours_reserve_both_per_thread_arrays():
    baseline, fixed = _calc_memory_bytes_non_local_means(
        (100, 200), np.dtype("float32")
    )
    selected, selected_fixed = _calc_memory_bytes_non_local_means(
        (100, 200), np.dtype("float32"), num_neighbours=10
    )
    assert selected >= baseline + 100 * 200 * 10 * 2 * 4
    assert selected_fixed == fixed


def test_nlm_small_images_bound_selection_by_processing_dimension():
    volume, _ = _calc_memory_bytes_non_local_means(
        (2, 2), np.dtype("float32"), num_neighbours=20
    )
    slices, _ = _calc_memory_bytes_non_local_means(
        (2, 2), np.dtype("float32"), num_neighbours=20, full_3d=False
    )
    assert volume > slices
    no_search, _ = _calc_memory_bytes_non_local_means(
        (2, 2), np.dtype("float32"), search_radius=0, num_neighbours=0
    )
    all_matches, _ = _calc_memory_bytes_non_local_means((2, 2), np.dtype("float32"))
    assert no_search == all_matches


@pytest.fixture
def nlm_memory_measurement():
    """Measure live pool allocations, including the input, on an isolated pool."""
    cp = pytest.importorskip("cupy")
    from httomolibgpu.cupywrapper import cupy_run
    from httomolibgpu.misc.denoise import non_local_means

    if not cupy_run:
        pytest.skip("CUDA GPU is required")

    class PeakMemoryHook(cp.cuda.MemoryHook):
        def __init__(self):
            self.current = 0
            self.peak = 0

        def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
            self.current += mem_size
            self.peak = max(self.peak, self.current)

        def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id):
            self.current -= mem_size

    def measure(shape, *, strided=False, **kwargs):
        per_slice, fixed = _calc_memory_bytes_non_local_means(
            shape[1:], np.dtype("float32"), **kwargs
        )
        estimated = shape[0] * per_slice + fixed
        free, _ = cp.cuda.runtime.memGetInfo()
        if estimated + 256 * 1024**2 > free:
            pytest.skip(
                f"Insufficient free GPU memory for {shape}: need {estimated} bytes plus headroom"
            )

        pool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(pool.malloc):
            # Compile and initialise the selected kernel before measuring data
            # allocations. CUDA context/compiler storage is outside this model.
            warmup = cp.ones((3, 3, 3), dtype=cp.float32)
            warm_result = non_local_means(warmup, **kwargs)
            cp.cuda.get_current_stream().synchronize()
            del warmup, warm_result
            pool.free_all_blocks()

            hook = PeakMemoryHook()
            with hook:
                storage_shape = (shape[0], shape[2], shape[1]) if strided else shape
                storage = cp.ones(storage_shape, dtype=cp.float32)
                data = storage.swapaxes(1, 2) if strided else storage
                result = non_local_means(data, **kwargs)
                cp.cuda.get_current_stream().synchronize()
                assert result.shape == shape
                assert result.dtype == cp.float32
                assert result.flags.c_contiguous
                del result, data, storage
            assert hook.current == 0
            pool.free_all_blocks()

        assert estimated >= hook.peak, (
            f"NLM memory underestimated for {shape}, {strided=}, {kwargs=}: "
            f"estimated {estimated}, measured {hook.peak} bytes"
        )
        return estimated, hook.peak

    return measure


@pytest.mark.cupy
@pytest.mark.parametrize("shape", [(8, 9, 11), (31, 64, 47), (64, 32, 128)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("full_3d", [False, True])
@pytest.mark.parametrize("num_neighbours", [0, 4])
@pytest.mark.parametrize(
    "search_radius, patch_radius", [(1, 0), (1, 1), (2, 1), (1, 2)]
)
def test_nlm_memory_scales_against_gpu_allocations(
    nlm_memory_measurement,
    shape,
    strided,
    full_3d,
    num_neighbours,
    search_radius,
    patch_radius,
    record_property,
):
    estimated_bytes, max_mem = nlm_memory_measurement(
        shape,
        strided=strided,
        full_3d=full_3d,
        search_radius=search_radius,
        patch_radius=patch_radius,
        num_neighbours=num_neighbours,
    )
    percentage_difference = 100 * (estimated_bytes - max_mem) / max_mem
    record_property("estimated_bytes", estimated_bytes)
    record_property("max_mem", max_mem)
    record_property("percentage_difference", percentage_difference)
    assert estimated_bytes >= max_mem
    # Selection arrays may reside in registers or CUDA local memory, which
    # CuPy allocation hooks do not measure. The model reserves them conservatively.
    if strided and num_neighbours == 0:
        assert percentage_difference <= 20, f"{percentage_difference=:.2f}%"


# Ten independent sizes for each axis: 1,000 shapes, up to 512 cubed.
# Opt in with --full; ordinary test runs retain the smaller cases above.
_NLM_DIMENSION_STEPS = tuple(int(n) for n in np.linspace(32, 512, 10))


@pytest.mark.full
@pytest.mark.cupy
@pytest.mark.parametrize("depth", _NLM_DIMENSION_STEPS)
@pytest.mark.parametrize("height", _NLM_DIMENSION_STEPS)
@pytest.mark.parametrize("width", _NLM_DIMENSION_STEPS)
def test_nlm_memory_scales_3d_shape_grid(
    nlm_memory_measurement, depth, height, width, record_property
):
    # Cycle radius pairs to keep 1,000 cases rather than another Cartesian
    # product. On large volumes, vary search and patch allocations separately;
    # simultaneous nonzero search and patch radii are covered by smaller cases.
    radius_pairs = ((1, 0), (2, 0), (0, 1), (0, 2))
    indices = (_NLM_DIMENSION_STEPS.index(size) for size in (depth, height, width))
    search_radius, patch_radius = radius_pairs[sum(indices) % len(radius_pairs)]
    estimated_bytes, max_mem = nlm_memory_measurement(
        (depth, height, width),
        strided=True,
        search_radius=search_radius,
        patch_radius=patch_radius,
        full_3d=True,
    )
    # Strided input exercises all three arrays reserved by the model. Start at
    # 32 cubed so the fixed buffer reserve does not dominate the percentage.
    percentage_difference = 100 * (estimated_bytes - max_mem) / max_mem
    record_property("search_radius", search_radius)
    record_property("patch_radius", patch_radius)
    record_property("estimated_bytes", estimated_bytes)
    record_property("max_mem", max_mem)
    record_property("percentage_difference", percentage_difference)
    assert estimated_bytes >= max_mem
    assert percentage_difference <= 20, (
        f"{(depth, height, width)=}, {search_radius=}, {patch_radius=}: "
        f"{estimated_bytes=}, {max_mem=}, {percentage_difference=:.2f}%"
    )
