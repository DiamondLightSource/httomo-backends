"""API contract tests; signature checks do not require a CUDA device."""

import importlib
from types import SimpleNamespace

import pytest
import yaml

from httomo_backends.methods_database.query import MethodsDatabaseQuery, Pattern
from httomo_backends.scripts import validate_api
from httomo_backends.scripts.yaml_templates_generator import (
    build_method_template,
    yaml_generator,
)


@pytest.mark.parametrize("package", validate_api.PACKAGES)
def test_installed_library_api(package):
    pytest.importorskip(package)
    assert validate_api.validate_package(package) == []


def test_cpu_normalization_registration():
    query = MethodsDatabaseQuery(
        "httomolib.prep.normalize", "dark_flat_field_correction"
    )
    assert query.get_pattern() == Pattern.projection
    assert query.get_implementation() == "cpu"
    assert query.get_output_dims_change() is False
    assert query.get_memory_gpu_params() is None
    assert query.padding() is False


@pytest.fixture
def contract(tmp_path, monkeypatch):
    """A small library whose API can change independently of its stored template."""

    def process(data, gain=2.0):
        return data

    module = "httomolib.prep.example"
    api = {module: {"process": process}}
    monkeypatch.setattr(validate_api, "_public_api", lambda package: api)
    base = tmp_path / "methods_database/packages/backends/httomolib"
    base.mkdir(parents=True)
    (base / "httomolib.yaml").write_text("prep:\n  example:\n    process: {}\n")
    (base / "httomolib_modules.yaml").write_text(f"- {module}\n")
    template = tmp_path / "yaml_templates/httomolib" / module / "process.yaml"
    template.parent.mkdir(parents=True)
    template.write_text(
        yaml.safe_dump(
            [
                {
                    "method": "process",
                    "module_path": module,
                    "parameters": {"gain": 2.0},
                }
            ]
        )
    )
    return SimpleNamespace(
        root=tmp_path, api=api, module=module, template=template, base=base
    )


def test_contract_accepts_matching_template(contract):
    assert validate_api.validate_package("httomolib", contract.root) == []


@pytest.mark.parametrize("change", ["default", "added_parameter", "removed_parameter"])
def test_contract_detects_signature_drift(contract, change):
    def changed_default(data, gain=3.0):
        pass

    def added_parameter(data, gain=2.0, clip=False):
        pass

    def removed_parameter(data):
        pass

    contract.api[contract.module]["process"] = {
        "default": changed_default,
        "added_parameter": added_parameter,
        "removed_parameter": removed_parameter,
    }[change]
    assert any(
        "template differs" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_contract_detects_new_public_module(contract):
    contract.api["httomolib.prep.new_module"] = {"new_method": lambda data: data}
    assert any(
        "not registered or explicitly deferred" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_contract_detects_removed_method(contract):
    contract.api[contract.module].clear()
    assert any(
        "registered method is not exported" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_contract_detects_missing_template(contract):
    contract.template.unlink()
    assert any(
        "missing YAML template" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_contract_detects_stale_template(contract):
    contract.template.with_name("removed.yaml").write_text("[]\n")
    assert any(
        "template has no registered method" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_contract_detects_missing_module_list_entry(contract):
    (contract.base / "httomolib_modules.yaml").write_text("[]\n")
    assert any(
        "module list and method database disagree" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_contract_detects_stale_pipeline_parameter(contract):
    pipeline = contract.root / "pipelines_full/example.yaml"
    pipeline.parent.mkdir()
    pipeline.write_text(
        yaml.safe_dump(
            [
                {
                    "module_path": contract.module,
                    "method": "process",
                    "parameters": {"removed": 1},
                }
            ]
        )
    )
    assert any(
        "has no parameter removed" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_contract_detects_stale_sweep_parameter(contract):
    pipeline = contract.root / "pipelines_full/example.yaml"
    pipeline.parent.mkdir()
    pipeline.write_text(
        yaml.safe_dump(
            [
                {
                    "module_path": contract.module,
                    "method": "process",
                    "sweep_parameter": "removed",
                }
            ]
        )
    )
    assert any(
        "has no parameter removed" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_generator_honours_output_argument(tmp_path, monkeypatch):
    def process(data, gain=2.0, *, clipping_warning=False, **options):
        pass

    module = SimpleNamespace(__all__=["process"], process=process)
    generator = importlib.import_module(
        "httomo_backends.scripts.yaml_templates_generator"
    )
    monkeypatch.setattr(generator.importlib, "import_module", lambda name: module)
    modules = tmp_path / "modules.yaml"
    modules.write_text("- example.prep.normalize\n")
    output = tmp_path / "chosen-output"
    assert yaml_generator(str(modules), str(output)) == 0
    template = yaml.safe_load(
        (output / "example.prep.normalize/process.yaml").read_text()
    )[0]
    assert template["parameters"] == {"gain": 2.0, "clipping_warning": False}


def test_generator_supported_only_excludes_unregistered_methods(tmp_path, monkeypatch):
    def process(data):
        pass

    module = SimpleNamespace(
        __all__=["process", "deferred"], process=process, deferred=process
    )
    generator = importlib.import_module(
        "httomo_backends.scripts.yaml_templates_generator"
    )
    monkeypatch.setattr(generator.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(
        generator, "_is_registered", lambda module, name: name == "process"
    )
    modules = tmp_path / "modules.yaml"
    modules.write_text("- example.prep.normalize\n")
    output = tmp_path / "templates"
    yaml_generator(str(modules), str(output), supported_only=True)
    assert sorted(path.name for path in output.rglob("*.yaml")) == ["process.yaml"]


def test_template_preserves_pipeline_conventions():
    def process(
        data, angles, center=None, axis=0, asynchronous=False, calc_peak_gpu_mem=False
    ):
        pass

    template = build_method_template("example.prep.normalize", "process", process)
    assert template["parameters"] == {
        "center": "${{centering.side_outputs.centre_of_rotation}}",
        "axis": "auto",
        "asynchronous": True,
    }


def test_contract_requires_explicit_deferred_method(contract, monkeypatch):
    name = "httomolib.prep.new_module.new_method"
    contract.api["httomolib.prep.new_module"] = {"new_method": lambda data: data}
    monkeypatch.setattr(
        validate_api, "DEFERRED_METHODS", {name: "Awaiting memory support."}
    )
    assert validate_api.validate_package("httomolib", contract.root) == []
    contract.api.pop("httomolib.prep.new_module")
    assert any(
        "stale deferred-method entry" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_contract_detects_registered_method_still_deferred(contract, monkeypatch):
    name = f"{contract.module}.process"
    monkeypatch.setattr(
        validate_api, "DEFERRED_METHODS", {name: "Old integration work."}
    )
    assert any(
        "still marked deferred" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )


def test_contract_detects_parameter_becoming_positional_only(contract):
    def positional_only(gain=2.0, /, data=None):
        pass

    contract.api[contract.module]["process"] = positional_only
    assert any(
        "template cannot bind" in error
        for error in validate_api.validate_package("httomolib", contract.root)
    )
