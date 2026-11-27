"""Check backend metadata and templates against installed library APIs.

Run ``python -m httomo_backends.scripts.validate_api`` in an environment containing
both HTTomolib and HTTomolibGPU. Imports inspect APIs without running GPU kernels.
"""

import argparse
from importlib import import_module
import inspect
from pathlib import Path
import pkgutil

import yaml

from httomo_backends.scripts.yaml_templates_generator import build_method_template

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("httomolib", "httomolibgpu")

# Public APIs awaiting the later memory/padding and alignment integration stages.
# Keep explicit reasons so newly exported methods cannot silently go unregistered.
DEFERRED_METHODS = {
    "httomolibgpu.recon.algorithm.iterative_alignment": (
        "Needs volume/shift side outputs and processing/memory support."
    ),
}


def _public_api(package):
    """Discover exported processing functions, including newly added modules."""
    library = import_module(package)
    api = {}
    for info in pkgutil.walk_packages(library.__path__, package + "."):
        relative = info.name.removeprefix(package + ".")
        if info.ispkg or relative.split(".")[0] not in {"misc", "prep", "recon"}:
            continue
        if any(part.startswith("_") for part in relative.split(".")):
            continue
        module = import_module(info.name)
        methods = {
            name: getattr(module, name)
            for name in getattr(module, "__all__", ())
            if inspect.isfunction(getattr(module, name))
        }
        if methods:
            api[info.name] = methods
    return api


def validate_package(package, root=ROOT):
    """Return actionable API, registration and template errors for one package."""
    base = root / "methods_database/packages/backends" / package
    database = yaml.safe_load((base / f"{package}.yaml").read_text())
    modules = yaml.safe_load((base / f"{package}_modules.yaml").read_text())
    api = _public_api(package)
    errors = []
    registered = {
        f"{package}.{category}.{module}": methods
        for category, groups in database.items()
        for module, methods in groups.items()
    }
    for module in sorted(set(modules) ^ set(registered)):
        errors.append(f"{module}: module list and method database disagree")
    for module in modules:
        if module not in api:
            errors.append(f"{module}: listed module exports no processing functions")
    public_names = {
        f"{module}.{name}" for module, methods in api.items() for name in methods
    }
    registered_names = {
        f"{module}.{name}" for module, methods in registered.items() for name in methods
    }
    deferred = {name for name in DEFERRED_METHODS if name.startswith(package + ".")}
    for name in sorted(public_names - registered_names - deferred):
        errors.append(f"{name}: public method is not registered or explicitly deferred")
    for name in sorted(deferred - public_names):
        errors.append(f"{name}: stale deferred-method entry")
    for name in sorted(deferred & registered_names):
        errors.append(f"{name}: registered method still marked deferred")

    template_root = root / "yaml_templates" / package
    expected_paths = set()
    for module, methods in registered.items():
        for name in methods:
            full_name = f"{module}.{name}"
            function = api.get(module, {}).get(name)
            if function is None:
                errors.append(f"{full_name}: registered method is not exported")
                continue
            path = template_root / module / f"{name}.yaml"
            expected_paths.add(path)
            if not path.exists():
                errors.append(f"{full_name}: missing YAML template")
                continue
            actual = yaml.safe_load(path.read_text())
            if isinstance(actual, list) and len(actual) == 1:
                try:
                    inspect.signature(function).bind_partial(
                        **(actual[0].get("parameters") or {})
                    )
                except TypeError as exc:
                    errors.append(f"{path}: template cannot bind to the API: {exc}")
            # YAML converts tuple defaults to lists; compare serialised values.
            try:
                expected = yaml.safe_load(
                    yaml.safe_dump([build_method_template(module, name, function)])
                )
            except ValueError as exc:
                errors.append(f"{path}: {exc}")
                continue
            if actual != expected:
                errors.append(
                    f"{path}: template differs from current API/defaults or HTTomo conventions"
                )
    for path in sorted(set(template_root.glob("*/*.yaml")) - expected_paths):
        errors.append(f"{path}: template has no registered method")

    # Directives may override defaults but may not refer to removed parameters.
    for path in sorted((root / "pipelines_full").glob("*.yaml")):
        for step in yaml.safe_load(path.read_text()) or []:
            module = step.get("module_path", "")
            if not module.startswith(package + "."):
                continue
            name = step["method"]
            if name not in registered.get(module, {}):
                errors.append(f"{path}: {module}.{name} is not registered")
                continue
            function = api.get(module, {}).get(name)
            if function is None:
                continue
            signature = inspect.signature(function)
            parameters = set((step.get("parameters") or {}).keys())
            if "sweep_parameter" in step:
                parameters.add(step["sweep_parameter"])
            for parameter in sorted(parameters - set(signature.parameters)):
                errors.append(f"{path}: {module}.{name} has no parameter {parameter}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=PACKAGES, action="append")
    args = parser.parse_args()
    errors = []
    for package in args.package or PACKAGES:
        library = import_module(package)
        print(f"Checking {package}: {library.__file__}")
        errors.extend(validate_package(package))
    for error in errors:
        print(error)
    if not errors:
        print(
            "Method registrations, templates and pipeline parameters match the installed APIs."
        )
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
