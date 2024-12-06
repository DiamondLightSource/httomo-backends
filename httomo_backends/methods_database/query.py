from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from types import ModuleType
from typing import Callable, List, Literal, Optional, Tuple
from pathlib import Path
import numpy as np

import yaml

YAML_DIR = Path(__file__).parent / "backends/"


class Pattern(Enum):
    """Enum for the different slicing-orientations/"patterns" that tomographic
    data can have.
    """

    projection = 0
    sinogram = 1
    all = 2


@dataclass(frozen=True)
class GpuMemoryRequirement:
    multiplier: Optional[float] = 1.0
    method: Literal["direct", "module"] = "direct"


def get_method_info(module_path: str, method_name: str, attr: str):
    """Get the information about the given method associated with `attr` that
    is stored in the relevant YAML file in `httomo/methods_database/packages/`

    Parameters
    ----------
    module_path : str
        The full module path of the method, including the top-level package
        name. Ie, `httomolib.misc.images.save_to_images`.

    method_name : str
        The name of the method function.

    attr : str
        The name of the piece of information about the method being requested
        (for example, "pattern").

    Returns
    -------
    The requested piece of information about the method.
    """
    method_path = f"{module_path}.{method_name}"
    split_method_path = method_path.split(".")
    package_name = split_method_path[0]

    # open the library file for the package
    ext_package_path = ""
    if package_name != "httomo":
        ext_package_path = f"{package_name}/"
    yaml_info_path = Path(YAML_DIR, str(ext_package_path), f"{package_name}.yaml")
    if not yaml_info_path.exists():
        err_str = f"The YAML file {yaml_info_path} doesn't exist."
        raise FileNotFoundError(err_str)

    with open(yaml_info_path, "r") as f:
        info = yaml.safe_load(f)
        for key in split_method_path[1:]:
            try:
                info = info[key]
            except KeyError:
                raise KeyError(f"The key {key} is not present ({method_path})")

    try:
        return info[attr]
    except KeyError:
        raise KeyError(f"The attribute {attr} is not present on {method_path}")
