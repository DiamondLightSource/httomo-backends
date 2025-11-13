from pathlib import Path

import pytest
import numpy as np

CUR_DIR = Path(__file__).parent


@pytest.fixture
def ensure_clean_memory():
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    yield None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "full: mark tests to run more GPU-memory consuming tests"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="run more GPU memory hungry tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        skip_other = pytest.mark.skip(reason="not a GPU hungry test")
        for item in items:
            if "full" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(
            reason="this GPU memory hungry test, use '--full' to run"
        )
        for item in items:
            if "full" in item.keywords:
                item.add_marker(skip_perf)


@pytest.fixture(scope="session")
def test_data_path():
    return CUR_DIR / "test_data"


# only load from disk once per session, and we use np.copy for the elements,
# to ensure data in this loaded file stays as originally loaded
@pytest.fixture(scope="session")
def data_file(test_data_path):
    in_file = test_data_path / "tomo_standard.npz"
    # keys: data, flats, darks, angles, angles_total, detector_y, detector_x
    return np.load(in_file)


@pytest.fixture
def flats(data_file, ensure_clean_memory):
    import cupy as cp

    return cp.asarray(data_file["flats"], dtype=cp.float32)


@pytest.fixture
def darks(data_file, ensure_clean_memory):
    import cupy as cp

    return cp.asarray(data_file["darks"], dtype=cp.float32)


@pytest.fixture
def data(data_file, ensure_clean_memory):
    import cupy as cp

    return cp.asarray(data_file["data"], dtype=cp.float32)


@pytest.fixture(scope="session")
def distortion_correction_path(test_data_path):
    return test_data_path / "distortion-correction"
