import contextlib
import importlib
import pkgutil
import sys


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


@contextlib.contextmanager
def fake_context():
    originals = {}

    submodules = list_submodules()
    for name, module in submodules.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        yield
    finally:
        for name, module in submodules.items():
            if originals[name] is None:
                del sys.modules[name]
            else:
                sys.modules[name] = originals[name]
