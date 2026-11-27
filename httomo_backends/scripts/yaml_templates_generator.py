#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Tomography Team <scientificsoftware@diamond.ac.uk>
# Created Date: 14/October/2022
# version ='0.3'
# ---------------------------------------------------------------------------
"""Script that exposes all functions of a given software package as YAML templates.

Please run the generator as:
    python -m httomo_backends.scripts.yaml_templates_generator -i /path/to/modules.yml -o /path/to/output/
"""

import argparse
import sys
import importlib
import inspect
from pathlib import Path
from typing import Any, List, Dict

import yaml

# Direct execution puts scripts/, rather than the checkout root, on sys.path.
# Prefer this checkout over an older installed copy of httomo_backends.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from httomo_backends.scripts._yaml_utils import (
    atomic_write,
    validate_name,
)  # noqa: E402


def yaml_generator(
    path_to_modules: str, output_folder: str, supported_only: bool = False
) -> int:
    """function that exposes all method of a given software package as YAML templates

    Args:
        path_to_modules: path to the list of modules yaml file
        output_folder: path to output folder with saved templates
        supported_only: generate only methods registered in the method database

    Returns:
        returns zero if the processing is succesfull
    """
    with open(path_to_modules, "r") as stream:
        modules_list = yaml.safe_load(stream)

    if not isinstance(modules_list, list) or not modules_list:
        raise ValueError("Module input must be a nonempty list of module names")
    for module_name in modules_list:
        validate_name(module_name, dotted=True)
    templates = []
    for module_name in modules_list:
        imported_module = importlib.import_module(module_name)
        exports = getattr(imported_module, "__all__", None)
        if not isinstance(exports, (list, tuple)):
            raise ValueError(f"{module_name}: expected an __all__ list of method names")
        for method_name in exports:
            validate_name(method_name)
            if supported_only and not _is_registered(module_name, method_name):
                continue
            method = getattr(imported_module, method_name)
            method_dict = build_method_template(module_name, method_name, method)
            # Serialise before touching any existing output files.
            text = yaml.safe_dump([method_dict], sort_keys=False)
            templates.append(
                (Path(output_folder) / module_name / f"{method_name}.yaml", text)
            )
    if supported_only and not templates:
        raise ValueError("No registered methods found in the requested modules")
    for path, text in templates:
        atomic_write(path, text)
    return 0


def _is_registered(module_name: str, method_name: str) -> bool:
    from httomo_backends.methods_database.query import MethodsDatabaseQuery

    try:
        MethodsDatabaseQuery(module_name, method_name).get_pattern()
    except (FileNotFoundError, KeyError):
        return False
    return True


def build_method_template(module_name: str, method_name: str, method) -> Dict:
    """Build a template from a callable, applying HTTomo parameter conventions."""
    validate_name(module_name, dotted=True)
    validate_name(method_name)
    if not callable(method):
        raise ValueError(f"{module_name}.{method_name}: export is not callable")
    params_dict: Dict[str, Any] = {}
    for name, value in inspect.signature(method).parameters.items():
        if name in _get_discard_keys() or value.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if value.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(
                f"{module_name}.{method_name}: positional-only parameter {name!r} cannot be a YAML keyword"
            )
        _set_param_value(name, method_name, value, params_dict)
    method_dict = {
        "method": method_name,
        "module_path": module_name,
        "parameters": params_dict,
    }
    _set_dict_special_cases(method_dict, method_name)
    return method_dict


def _set_param_value(
    name: str, method_name: str, value: inspect.Parameter, params_dict: Dict[str, Any]
):
    """Set param value for method inside dictionary
    Args:
        name: Parameter name
        method_name: method's name
        value: Parameter value
        params_dict: Dict containing method's parameter names and values
    """
    if value.default is inspect.Parameter.empty:
        if name in ["proj1", "proj2"]:
            params_dict[name] = "auto"
        else:
            params_dict[name] = "REQUIRED"
    elif name == "axis":
        params_dict[name] = "auto"
    elif name == "asynchronous":
        params_dict[name] = True
    elif name == "center":
        # Temporary value
        params_dict[name] = "${{centering.side_outputs.centre_of_rotation}}"
    elif name == "glob_stats":
        params_dict[name] = "${{statistics.side_outputs.glob_stats}}"
    elif name == "overlap":
        params_dict[name] = "${{centering.side_outputs.overlap}}"
    elif name == "side" and method_name == "sino_360_to_180":
        params_dict[name] = "${{centering.side_outputs.side}}"
    else:
        params_dict[name] = value.default


def _save_yaml(
    module_name: str, method_name: str, params_list: List[Dict], output_folder: str
):
    """Save the list as a YAML file
    Args:
        module_name: Name of module
        method_name: Name of method
        params_list: List of parameters
    """
    path = Path(output_folder) / module_name / f"{method_name}.yaml"
    atomic_write(path, yaml.safe_dump(params_list, sort_keys=False))


def _set_dict_special_cases(method_dict: Dict, method_name: str):
    """Dealing with special cases for "data_out"

    Args:
        method_dict: Dictionary of modules and parameters
        method_name: Name of method
    """
    if method_name in [
        "find_center",
        "find_center_vo",
        "find_center_pc",
        "find_center_metric_recon",
    ]:
        method_dict["id"] = "centering"
        method_dict["side_outputs"] = {"cor": "centre_of_rotation"}
    if method_name in ["find_center_360"]:
        method_dict["id"] = "centering"
        method_dict["side_outputs"] = {
            "cor": "centre_of_rotation",
            "overlap": "overlap",
            "side": "side",
            "overlap_position": "overlap_position",
        }
    if method_name in ["calculate_stats"]:
        method_dict["id"] = "statistics"
        method_dict["side_outputs"] = {"glob_stats": "glob_stats"}


def _get_discard_data_out() -> List[str]:
    """Discard data_out from certain modules

    Returns: list of data_out to discard
    """
    discard_data_out = ["save_to_images"]
    return discard_data_out


def _get_discard_keys() -> List[str]:
    """Can work with any software in principle,
    but for TomoPy, httomolib(gpu) there are additional keys
    that needed to be discarded in templates in order to let
    httomo work smoothly.

    Returns: List of keys to discard
    """
    discard_keys = [
        "in_file",
        "data_in",
        "tomo",
        "arr",
        "prj",
        "data",
        "ncore",
        "nchunk",
        "flats",
        "flat",
        "dark",
        "darks",
        "theta",
        "out",
        "ang",
        "comm_rank",
        "out_dir",
        "angles",
        "gpu_id",
        "comm",
        "offset",
        "shift_xy",
        "step_xy",
        "jpeg_quality",
        "watermark_vals",
        "power_of_2_oversampling",
        "power_of_2_cropping",
        "min_mem_usage_filter",
        "min_mem_usage_ifft2",
        "calc_peak_gpu_mem",
        "shift_seam_index",
        "bit_depth",
    ]
    return discard_keys


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that exposes all functions "
        "of a given software package as YAML templates."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="A path to the list of modules yaml file"
        "which is needed to be inspected and functions extracted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="Directory to save the yaml templates in.",
    )
    parser.add_argument(
        "--supported-only",
        action="store_true",
        help="Generate only methods registered in the method database.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    path_to_modules = args.input
    output_folder = args.output
    return_val = yaml_generator(path_to_modules, output_folder, args.supported_only)
    if return_val == 0:
        print("The methods as YAML templates have been successfully generated!")
