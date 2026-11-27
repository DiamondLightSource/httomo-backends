"""CPU checks for the nonlinear diffusion scheduling estimate."""

import numpy as np
import pytest

from httomo_backends.methods_database.query import MethodsDatabaseQuery, Pattern


def query():
    return MethodsDatabaseQuery("httomolibgpu.misc.denoise", "nonlinear_diffusion")


def test_ndf_uses_module_estimator():
    memory = query().get_memory_gpu_params()
    assert memory.method == "module"
    assert memory.multiplier is None


def test_ndf_rejects_half_precision():
    with pytest.raises(ValueError, match="half_precision=False"):
        query().calculate_memory_bytes(
            (100, 200), np.dtype("float32"), half_precision=True
        )


def test_ndf_memory_scales_with_slice_area():
    small, fixed = query().calculate_memory_bytes((100, 200), np.dtype("float32"))
    large, large_fixed = query().calculate_memory_bytes((200, 200), np.dtype("float32"))
    assert large == 2 * small
    assert fixed == large_fixed


def test_ndf_default_matches_explicit_full_precision():
    default = query().calculate_memory_bytes((100, 200), np.dtype("float32"))
    explicit = query().calculate_memory_bytes(
        (100, 200), np.dtype("float32"), half_precision=False
    )
    assert default == explicit


def test_ndf_registration_and_padding():
    method = query()
    assert method.get_pattern() == Pattern.all
    assert method.get_implementation() == "gpu_cupy"
    assert method.get_output_dims_change() is False
    assert method.save_result_default() is False
    assert method.padding() is True
    assert method.calculate_padding() == (5, 5)


def test_ndf_pipeline_generation(tmp_path):
    import yaml
    from pathlib import Path
    from httomo_backends.scripts.yaml_pipelines_generator import (
        yaml_pipelines_generator,
    )

    root = Path(__file__).resolve().parents[1] / "httomo_backends"
    directive = tmp_path / "directive.yaml"
    directive.write_text(
        "- module_path: httomolibgpu.misc.denoise\n"
        "  method: nonlinear_diffusion\n"
        "  parameters:\n"
        "    half_precision: false\n"
    )
    output = tmp_path / "pipeline.yaml"
    yaml_pipelines_generator(directive, root, output)
    method = yaml.safe_load(output.read_text())[0]
    assert method["method"] == "nonlinear_diffusion"
    assert method["parameters"]["half_precision"] is False
    assert method["parameters"]["iterations"] == 150
    assert method["parameters"]["sigma"] == 0.01
    assert method["parameters"]["time_marching_parameter"] == 0.005
    assert method["parameters"]["penalty_type"] == 1
