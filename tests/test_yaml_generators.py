"""Generation tests use small templates and do not import reconstruction libraries."""

from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import ruamel.yaml
import yaml

from httomo_backends.scripts import yaml_templates_generator as templates
from httomo_backends.scripts.yaml_pipelines_generator import yaml_pipelines_generator

ROOT = Path(__file__).resolve().parents[1] / "httomo_backends"


@pytest.fixture
def pipeline(tmp_path):
    template = tmp_path / "backend/yaml_templates/example/example.misc.process/run.yaml"
    template.parent.mkdir(parents=True)
    template.write_text(
        yaml.safe_dump(
            [
                {
                    "method": "run",
                    "module_path": "example.misc.process",
                    "parameters": {"strength": 1.0},
                }
            ]
        )
    )
    return SimpleNamespace(
        template=template,
        backend=tmp_path / "backend",
        directive=tmp_path / "directive.yaml",
        output=tmp_path / "results/pipeline.yaml",
    )


def generate(pipeline, steps):
    pipeline.directive.write_text(yaml.safe_dump(steps))
    yaml_pipelines_generator(pipeline.directive, pipeline.backend, pipeline.output)
    return ruamel.yaml.YAML(typ="rt").load(pipeline.output)


def step(**kwargs):
    return {"module_path": "example.misc.process", "method": "run", **kwargs}


def test_pipeline_preserves_overrides_and_metadata(pipeline):
    output = generate(
        pipeline, [step(parameters={"strength": 0.0}, id="result", save_result=False)]
    )
    assert output[0]["parameters"]["strength"] == 0.0
    assert output[0]["id"] == "result"
    assert output[0]["save_result"] is False


def test_pipeline_repeated_method_overrides_are_independent(pipeline):
    output = generate(pipeline, [step(parameters={"strength": 3.0}), step()])
    assert [item["parameters"]["strength"] for item in output] == [3.0, 1.0]


@pytest.mark.parametrize(
    "sweep,tag",
    [
        ({"sweep_values": [1, 2]}, "!Sweep"),
        ({"sweep_start": 0, "sweep_stop": 2, "sweep_step": 0.5}, "!SweepRange"),
        ({"sweep_start": 2, "sweep_stop": 0, "sweep_step": -1}, "!SweepRange"),
    ],
)
def test_sweeps_are_local_to_their_step(pipeline, sweep, tag):
    output = generate(pipeline, [step(sweep_parameter="strength", **sweep), step()])
    assert str(output[0]["parameters"]["strength"].tag) == tag
    assert output[1]["parameters"]["strength"] == 1.0


@pytest.mark.parametrize(
    "changes",
    [
        {"sweep_parameter": "removed", "sweep_values": [1]},
        {"sweep_parameter": "strength", "sweep_values": []},
        {"sweep_parameter": "strength", "sweep_values": "123"},
        {"sweep_parameter": "strength", "sweep_start": 0},
        {"sweep_values": [1]},
        {
            "sweep_parameter": "strength",
            "sweep_start": 0,
            "sweep_stop": 1,
            "sweep_step": 0,
        },
        {
            "sweep_parameter": "strength",
            "sweep_start": 1,
            "sweep_stop": 0,
            "sweep_step": 1,
        },
        {
            "sweep_parameter": "strength",
            "sweep_start": 1,
            "sweep_stop": 1,
            "sweep_step": 1,
        },
        {
            "sweep_parameter": "strength",
            "sweep_start": 0,
            "sweep_stop": float("nan"),
            "sweep_step": 1,
        },
        {
            "sweep_parameter": "strength",
            "sweep_start": False,
            "sweep_stop": 1,
            "sweep_step": 1,
        },
        {
            "sweep_parameter": "strength",
            "sweep_start": 0,
            "sweep_stop": 1,
            "sweep_step": 1,
            "sweep_values": [1],
        },
        {"parameters": []},
        {"parameters": {"removed": 1}},
        {"unknown_field": 1},
        {"module_path": "../../outside"},
        {"method": "../outside"},
    ],
)
def test_invalid_directives_preserve_existing_output(pipeline, changes):
    pipeline.output.parent.mkdir()
    pipeline.output.write_text("existing output\n")
    with pytest.raises(ValueError):
        generate(pipeline, [step(**changes)])
    assert pipeline.output.read_text() == "existing output\n"


@pytest.mark.parametrize("content", ["", "[]", "{}", "- invalid", "- method: run"])
def test_pipeline_rejects_invalid_top_level(pipeline, content):
    pipeline.directive.write_text(content)
    with pytest.raises(ValueError):
        yaml_pipelines_generator(pipeline.directive, pipeline.backend, pipeline.output)
    assert not pipeline.output.exists()


def test_missing_template_preserves_existing_output(pipeline):
    pipeline.output.parent.mkdir()
    pipeline.output.write_text("existing output\n")
    with pytest.raises(FileNotFoundError):
        generate(pipeline, [step(), step(method="missing")])
    assert pipeline.output.read_text() == "existing output\n"


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "- method: other\n  parameters: {}",
        "- method: run\n  module_path: example.misc.process\n  parameters: []",
    ],
)
def test_pipeline_rejects_invalid_template(pipeline, content):
    pipeline.template.write_text(content)
    with pytest.raises(ValueError, match="expected one matching method"):
        generate(pipeline, [step()])
    assert not pipeline.output.exists()


def test_rotation_without_side_outputs_does_not_crash(tmp_path):
    directive = tmp_path / "directive.yaml"
    directive.write_text(
        "- module_path: tomopy.recon.rotation\n  method: write_center\n"
    )
    output = tmp_path / "pipeline.yaml"
    yaml_pipelines_generator(directive, ROOT, output)
    assert yaml.safe_load(output.read_text())[0]["method"] == "write_center"


def test_tomopy_algorithm_is_not_overwritten(tmp_path):
    directive = tmp_path / "directive.yaml"
    directive.write_text(
        "- module_path: tomopy.recon.algorithm\n  method: recon\n  parameters:\n    algorithm: sirt\n"
    )
    output = tmp_path / "pipeline.yaml"
    yaml_pipelines_generator(directive, ROOT, output)
    assert yaml.safe_load(output.read_text())[0]["parameters"]["algorithm"] == "sirt"


@pytest.mark.parametrize(
    "path", sorted((ROOT / "pipelines_full").glob("*.yaml")), ids=lambda path: path.name
)
def test_bundled_pipeline_directives_generate(path, tmp_path):
    output = tmp_path / "pipeline.yaml"
    yaml_pipelines_generator(path, ROOT, output)
    generated = ruamel.yaml.YAML(typ="rt").load(output)
    directives = yaml.safe_load(path.read_text())
    assert [(item["module_path"], item["method"]) for item in generated] == [
        (item["module_path"], item["method"]) for item in directives
    ]


@pytest.mark.parametrize("content", ["", "[]", "{}", "name", "- 1", "- ../outside"])
def test_template_generator_rejects_invalid_module_lists(tmp_path, content):
    modules = tmp_path / "modules.yaml"
    modules.write_text(content)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        templates.yaml_generator(modules, output)
    assert not output.exists()


def test_template_generator_failure_does_not_replace_previous_output(
    tmp_path, monkeypatch
):
    def process(data, gain=1):
        pass

    def invalid(data, option=object()):
        pass

    module = SimpleNamespace(
        __all__=["process", "invalid"], process=process, invalid=invalid
    )
    monkeypatch.setattr(templates.importlib, "import_module", lambda name: module)
    modules = tmp_path / "modules.yaml"
    modules.write_text("- example.prep.normalize\n")
    output = tmp_path / "output/example.prep.normalize/process.yaml"
    output.parent.mkdir(parents=True)
    output.write_text("existing template\n")
    with pytest.raises(yaml.representer.RepresenterError):
        templates.yaml_generator(modules, tmp_path / "output")
    assert output.read_text() == "existing template\n"


def test_template_generator_rejects_positional_only_configuration():
    def process(strength=1, /, data=None):
        pass

    with pytest.raises(ValueError, match="positional-only parameter"):
        templates.build_method_template("example.prep.normalize", "process", process)


def test_regular_parameter_named_kwargs_is_not_lost():
    def process(data, kwargs=None):
        pass

    result = templates.build_method_template(
        "example.prep.normalize", "process", process
    )
    assert result["parameters"] == {"kwargs": None}


def test_representer_registration_does_not_change_other_yaml_instances(pipeline):
    representers = dict(ruamel.yaml.representer.RoundTripRepresenter.yaml_representers)
    generate(pipeline, [step(sweep_parameter="strength", sweep_values=[1, 2])])
    assert (
        ruamel.yaml.representer.RoundTripRepresenter.yaml_representers == representers
    )


def test_tomopy_phase_has_no_incorrect_paganin_unit_comments(tmp_path):
    directive = tmp_path / "directive.yaml"
    directive.write_text("- module_path: tomopy.prep.phase\n  method: retrieve_phase\n")
    output = tmp_path / "pipeline.yaml"
    yaml_pipelines_generator(directive, ROOT, output)
    assert "MICRON" not in output.read_text()
    assert "METRE" not in output.read_text()


@pytest.mark.parametrize(
    "module_name,arguments",
    [
        ("yaml_templates_generator", []),
        ("yaml_pipelines_generator", []),
        ("yaml_pipelines_generator", ["-i", "directives.yaml"]),
    ],
)
def test_cli_reports_missing_required_paths(monkeypatch, module_name, arguments):
    import importlib
    import sys

    module = importlib.import_module(f"httomo_backends.scripts.{module_name}")
    monkeypatch.setattr(sys, "argv", ["generator", *arguments])
    with pytest.raises(SystemExit) as error:
        module.get_args()
    assert error.value.code == 2


def test_supported_only_reports_no_matching_methods(tmp_path, monkeypatch):
    def process(data):
        pass

    monkeypatch.setattr(
        templates.importlib,
        "import_module",
        lambda name: SimpleNamespace(__all__=["process"], process=process),
    )
    monkeypatch.setattr(templates, "_is_registered", lambda module, name: False)
    modules = tmp_path / "modules.yaml"
    modules.write_text("- example.prep.normalize\n")
    with pytest.raises(ValueError, match="No registered methods"):
        templates.yaml_generator(modules, tmp_path / "output", supported_only=True)
    assert not (tmp_path / "output").exists()


def test_template_script_runs_by_file_path_from_other_directory(tmp_path):
    # A separate process reproduces script sys.path ordering and installed-package conflicts.
    module = tmp_path / "example_backend.py"
    module.write_text(
        "__all__ = ['process']\ndef process(data, gain=2):\n    return data\n"
    )
    modules = tmp_path / "modules.yaml"
    modules.write_text("- example_backend\n")
    output = tmp_path / "templates"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/yaml_templates_generator.py"),
            "-i",
            str(modules),
            "-o",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    template = yaml.safe_load((output / "example_backend/process.yaml").read_text())
    assert template[0]["parameters"] == {"gain": 2}


def test_pipeline_script_runs_by_file_path_from_other_directory(tmp_path):
    directive = tmp_path / "directives.yaml"
    directive.write_text(
        "- module_path: httomolib.misc.rescale\n  method: rescale_to_int\n"
    )
    output = tmp_path / "pipeline.yaml"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/yaml_pipelines_generator.py"),
            "-i",
            str(directive),
            "-o",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        yaml.safe_load(output.read_text())[0]["module_path"] == "httomolib.misc.rescale"
    )
