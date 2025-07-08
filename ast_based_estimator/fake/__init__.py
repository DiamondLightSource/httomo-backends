from . import cupy
from . import cupyx

import os
import ast
import importlib.util


def __find_module_path():
    spec = importlib.util.find_spec(__package__ or __name__.split(".")[0])
    if spec is None or spec.origin is None:
        raise ImportError("Module not found")

    if spec.submodule_search_locations:
        return spec.submodule_search_locations[0]  # it's a package
    else:
        return os.path.dirname(spec.origin)  # it's a single file module


def __find_py_files(package_path):
    py_files = []
    for root, _, files in os.walk(package_path):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


def __list_defs_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        node = ast.parse(f.read(), filename=file_path)

    functions = []
    classes = []

    for item in node.body:
        if (
            isinstance(item, ast.FunctionDef)
            and not item.name.startswith("_")
            and item.name != "top_level_definitions"
        ):
            functions.append(item.name)
        elif isinstance(item, ast.ClassDef) and not item.name.startswith("_"):
            classes.append(item.name)

    return functions, classes


def top_level_definitions():
    package_path = __find_module_path()
    py_files = __find_py_files(package_path)
    results = set()
    for py_file in py_files:
        functions, classes = __list_defs_from_file(py_file)
        results.update(functions, classes)
    return results
