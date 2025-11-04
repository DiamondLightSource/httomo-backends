import contextlib
import importlib
import pkgutil
import sys
import types


def list_submodules():
    package = importlib.import_module(__name__)
    submodule_names = [
        name
        for _, name, _ in pkgutil.walk_packages(
            package.__path__, prefix=package.__name__ + "."
        )
    ]
    return {
        name.removeprefix(package.__name__ + "."): importlib.import_module(name)
        for name in submodule_names
    }


def recursive_reload(module):
    importlib.reload(module)

    if hasattr(module, "__path__"):
        prefix = f"{module.__name__}."
        cupywrapper_name = f"{prefix}cupywrapper"

        package_names = [x.name for x in pkgutil.walk_packages(module.__path__, prefix)]
        package_names = (
            [cupywrapper_name] + [x for x in package_names if x != cupywrapper_name]
            if cupywrapper_name in package_names
            else package_names
        )
        print(f"********* package_names: {package_names}")

        for name in package_names:
            if name in sys.modules:
                print(f"reloading {name}")
                recursive_reload(sys.modules[name])


@contextlib.contextmanager
def fake_context(*modules_to_reload: types.ModuleType):
    memory_usage = {"peak_memory": 0, "current_peak_memory": 0}
    fft_plan_cache = {}

    globals()["memory_usage"] = memory_usage
    globals()["fft_plan_cache"] = fft_plan_cache

    originals = {}

    submodules = list_submodules()
    for name, module in submodules.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    for module in modules_to_reload:
        recursive_reload(module)

    try:
        yield
    finally:
        del globals()["memory_usage"]
        del globals()["fft_plan_cache"]

        for name, module in submodules.items():
            if originals[name] is None:
                del sys.modules[name]
            else:
                sys.modules[name] = originals[name]

        for module in modules_to_reload:
            recursive_reload(module)
