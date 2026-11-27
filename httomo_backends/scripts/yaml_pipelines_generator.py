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
# Created Date: 22/January/2025
# version ='0.1'
# ---------------------------------------------------------------------------
"""Script that generates YAML pipeline for HTTomo using YAML templates from httomo-backends
(should be already installed in your environment).

Please run the generator as:
    python -m httomo_backends.scripts.yaml_pipelines_generator -i directives.yaml -o pipeline.yaml
"""

import argparse
import sys
import os
from pathlib import Path
from numbers import Real
import math
from io import StringIO
import ruamel.yaml
import yaml

# Direct execution puts scripts/, rather than the checkout root, on sys.path.
# Prefer this checkout over an older installed copy of httomo_backends.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httomo_backends  # noqa: E402

from httomo_backends.scripts._yaml_utils import (
    atomic_write,
    validate_name,
)  # noqa: E402

CS = ruamel.yaml.comments.CommentedSeq  # defaults to block style


class SweepRange:
    """SweepRange class."""

    def __init__(self, start, stop, step):
        self._start, self._stop, self._step = start, stop, step


def __sweeprange_representer(
    dumper: yaml.SafeDumper, swp: SweepRange
) -> yaml.nodes.MappingNode:
    """Represent a sweeprange as a YAML mapping node."""
    return dumper.represent_mapping(
        "!SweepRange",
        {
            "start": swp._start,
            "stop": swp._stop,
            "step": swp._step,
        },
    )


class SweepManual:
    """SweepManual class."""

    def __init__(self, lst):
        self._lst = lst


def __sweepmanual_representer(
    dumper: yaml.SafeDumper, swp: SweepManual
) -> yaml.nodes.SequenceNode:
    """Represent a sweepmanual as a YAML sequence node."""
    return dumper.represent_sequence("!Sweep", swp._lst)


def __represent_none(self, data):
    return self.represent_scalar("tag:yaml.org,2002:null", "null")


def yaml_pipelines_generator(
    path_to_pipelines: str, path_to_httomobackends: str, path_to_output_file: str
) -> int:
    """function that builds YAML pipeline using YAML templates from httomo-backends

    Args:
        path_to_pipelines: path to the YAML file which contains a high-level description of the required pipeline to be built.
        path_to_httomobackends: path to httomo-backends on the system, where YAML templates stored.
        path_to_output_file: path to output file with the generated pipeline

    Returns:
        returns zero if the processing is successful
    """

    yaml = ruamel.yaml.YAML(typ="rt", pure=True)

    # Keep custom representers local to this generator instance.
    class PipelineRepresenter(ruamel.yaml.representer.RoundTripRepresenter):
        pass

    yaml.Representer = PipelineRepresenter
    with open(path_to_pipelines) as stream:
        pipeline_file_content = yaml.load(stream)
    if not isinstance(pipeline_file_content, list) or not pipeline_file_content:
        raise ValueError("Pipeline directives must be a nonempty list of methods")
    allowed = {
        "method",
        "module_path",
        "parameters",
        "id",
        "side_outputs",
        "save_result",
        "sweep_parameter",
        "sweep_start",
        "sweep_stop",
        "sweep_step",
        "sweep_values",
    }
    for index, step in enumerate(pipeline_file_content, 1):
        if not isinstance(step, dict) or not {"method", "module_path"} <= step.keys():
            raise ValueError(f"Step {index}: method and module_path are required")
        unknown = set(step) - allowed
        if unknown:
            raise ValueError(
                f"Step {index}: unknown directive fields: {sorted(unknown, key=str)}"
            )
    # a loop over methods in the high-level pipeline file (directive)
    methods_no = len(pipeline_file_content)
    pipeline_full = CS()
    for i in range(methods_no):
        method_content = pipeline_file_content[i]
        method_name = method_content["method"]
        module_name = method_content["module_path"]
        validate_name(module_name, dotted=True)
        validate_name(method_name)
        template_path = (
            Path(path_to_httomobackends)
            / "yaml_templates"
            / module_name.split(".")[0]
            / module_name
            / f"{method_name}.yaml"
        )
        with template_path.open() as stream:
            yaml_template_method = yaml.load(stream)
        if (
            not isinstance(yaml_template_method, list)
            or len(yaml_template_method) != 1
            or not isinstance(yaml_template_method[0], dict)
            or yaml_template_method[0].get("method") != method_name
            or yaml_template_method[0].get("module_path") != module_name
            or not isinstance(yaml_template_method[0].get("parameters"), dict)
        ):
            raise ValueError(
                f"{template_path}: expected one matching method with a parameter mapping"
            )
        overrides = method_content.get("parameters", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Step {i + 1}: parameters must be a mapping")
        unknown = set(overrides) - set(yaml_template_method[0]["parameters"])
        if unknown:
            raise ValueError(
                f"Step {i + 1}: parameters not present in the template: {sorted(unknown, key=str)}"
            )

        pipeline_full.yaml_set_start_comment(
            "This pipeline should be supported by the latest developments of HTTomo. Use module load httomo/latest module at Diamond."
        )

        if "loaders" in module_name:
            # should be the first method in the list
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Standard tomography loader for NeXus files. ---",
                indent=0,
            )
            pipeline_full += yaml_template_method
        elif "rotation" in module_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Center of Rotation auto-finding. Required for reconstruction bellow. ---",
                indent=0,
            )
            pipeline_full += yaml_template_method
            _add_comment(
                pipeline_full[i]["parameters"],
                key="ind",
                comment="A vertical slice (sinogram) index to calculate CoR, 'mid' can be used for middle",
            )
            _add_comment(
                pipeline_full[i]["parameters"],
                key="cor_initialisation_value",
                comment="Use if an approximate CoR is known",
            )
            _add_comment(
                pipeline_full[i]["parameters"],
                key="average_radius",
                comment="Average several sinograms to improve SNR, one can try 3-5 range",
            )
            _add_comment(
                pipeline_full[i]["parameters"],
                key="side",
                comment="Choose 'left' to stitch to the left side, 'right' to the right side, or null for automated determination",
            )
            _add_comment(
                pipeline_full[i].get("side_outputs", {}),
                key="cor",
                comment="An estimated CoR value provided as a side output",
            )
            _add_comment(
                pipeline_full[i].get("side_outputs", {}),
                key="overlap",
                comment="An overlap to use for converting 360 degrees scan to 180 degrees scan.",
            )
        elif "corr" in module_name and "remove_outlier" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Removing unresponsive/dead pixels in the data, aka zingers. Use if sharp streaks are present in the reconstruction. To be applied before normalisation. ---",
                indent=0,
            )
            pipeline_full += yaml_template_method
            if pipeline_full[i]["parameters"].get("dif") == "REQUIRED":
                # fix for the absent parameter in TomoPy's algorithm
                pipeline_full[i]["parameters"]["dif"] = 0.1
            _add_comment(
                pipeline_full[i]["parameters"],
                key="kernel_size",
                comment="The size of the 3D neighbourhood surrounding the voxel. Odd integer.",
            )
            _add_comment(
                pipeline_full[i]["parameters"],
                key="dif",
                comment="A difference between the outlier value and the median value of neighbouring pixels.",
            )

        elif "distortion" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Applying optical distortion correction to projections. --- ",
                indent=0,
            )
            pipeline_full += yaml_template_method
            _add_comment(
                pipeline_full[i]["parameters"],
                key="metadata_path",
                comment="Provide an absolute path to the text file with distortion coefficients.",
            )
        elif "seam_blend" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Applying seam blending with ramp filter after data is stitched. --- ",
                indent=0,
            )
            pipeline_full += yaml_template_method
            _add_comment(
                pipeline_full[i]["parameters"],
                key="path_to_stiched_params_file",
                comment="Provide an absolute path to the text file with seam index and blending width.",
            )
        elif "average_projection_frames" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Apply averaging of projection data in the angular dimension. --- ",
                indent=0,
            )
            pipeline_full += yaml_template_method
        elif "data_resampler" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Down/up sampling the data. --- ",
                indent=0,
            )
            pipeline_full += yaml_template_method
            _add_comment(
                pipeline_full[i]["parameters"],
                key="newshape",
                comment="Provide a new shape for a 2D slice, e.g. [256, 256].",
            )
        elif "sino_360_to_180" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Using the overlap and side provided, converting 360 degrees scan to 180 degrees scan. --- ",
                indent=0,
            )
            pipeline_full += yaml_template_method
        elif "dark_flat_field_correction" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Flat-field and dark-field projection correction. --- ",
                indent=0,
            )
            pipeline_full += yaml_template_method
        elif "minus_log" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Negative log is required for reconstruction to convert raw intensity measurements into the line integrals of attenuation. --- ",
                indent=0,
            )
            pipeline_full += yaml_template_method
        elif "phase" in module_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Apply a phase contrast filter to improve image contrast. --- ",
                indent=0,
            )
            pipeline_full += yaml_template_method
            if (
                module_name in {"httomolib.prep.phase", "httomolibgpu.prep.phase"}
                and method_name == "paganin_filter"
            ):
                _add_comment(
                    pipeline_full[i]["parameters"],
                    key="ratio_delta_beta",
                    comment="The ratio of delta/beta for filter strength control. Larger values lead to more smoothing.",
                )
                _add_comment(
                    pipeline_full[i]["parameters"],
                    key="pixel_size",
                    comment="Detector pixel size (resolution) in MICRON units.",
                )
                _add_comment(
                    pipeline_full[i]["parameters"],
                    key="distance",
                    comment="Propagation distance of the wavefront from sample to detector in METRE units.",
                )
                _add_comment(
                    pipeline_full[i]["parameters"],
                    key="energy",
                    comment="Beam energy in keV.",
                )
                _add_comment(
                    pipeline_full[i]["parameters"],
                    key="calculate_padding_value_method",
                    comment="Select type of padding from 'next_power_of_2', 'next_fast_length' and 'use_pad_x_y'.",
                )
                _add_comment(
                    pipeline_full[i]["parameters"],
                    key="pad_x_y",
                    comment="Manual padding is enabled when 'calculate_padding_value_method' is set to 'use_pad_x_y'.",
                )
        elif "stripe" in module_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Method to remove stripe artefacts in the data that lead to ring artefacts in the reconstruction. --- ",
                indent=0,
            )
            pipeline_full += yaml_template_method
        elif "algorithm" in module_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Reconstruction method. ---",
                indent=0,
            )
            pipeline_full += yaml_template_method
            _add_comment(
                pipeline_full[i]["parameters"],
                key="center",
                comment="Reference to center of rotation side output, a float number or 'null' for a middle of the horizontal dimension.",
            )
            _add_comment(
                pipeline_full[i]["parameters"],
                key="detector_pad",
                comment="Horizontal detector padding to minimise circle/arc-type artifacts in the reconstruction. Set to 'true' to enable automatic padding or an integer",
            )
            _add_comment(
                pipeline_full[i]["parameters"],
                key="recon_mask_radius",
                comment="Zero pixels outside the mask-circle radius. Make radius equal to 2.0 to remove the mask effect.",
            )
            _add_comment(
                pipeline_full[i]["parameters"],
                key="neglog",
                comment="Perform negative log here if it was previously switched off.",
            )
            if pipeline_full[i]["parameters"].get("algorithm", "absent") is None:
                # fix for a default parameter (None) in TomoPy's algorithm
                pipeline_full[i]["parameters"]["algorithm"] = "gridrec"
                _add_comment(
                    pipeline_full[i]["parameters"],
                    key="algorithm",
                    comment="Select the required algorithm, e.g. 'gridrec'",
                )
        elif "denoise" in module_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Using denoising method to reduce noise. ---",
                indent=0,
            )
            pipeline_full += yaml_template_method
        elif "calculate_stats" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Calculate global statistics on the reconstructed volume, required for data rescaling. ---",
                indent=0,
            )
            pipeline_full += yaml_template_method
        elif "rescale_to_int" in method_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Rescaling the data using min/max obtained from `calculate_stats`. ---",
                indent=0,
            )
            pipeline_full += yaml_template_method
        elif "images" in module_name:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--- Saving data into images. ---",
                indent=0,
            )
            pipeline_full += yaml_template_method
            _add_comment(
                pipeline_full[i]["parameters"],
                key="file_format",
                comment="`tif` or `jpeg` can be used.",
            )
        else:
            pipeline_full.yaml_set_comment_before_after_key(
                i,
                "--------------------------------------------------------#",
                indent=0,
            )
            pipeline_full += yaml_template_method

        pipeline_full[i]["parameters"].update(overrides)
        for key in ("id", "side_outputs", "save_result"):
            if key in method_content:
                pipeline_full[i][key] = method_content[key]
        _set_sweep(pipeline_full[i]["parameters"], method_content, i + 1)

    yaml.representer.add_representer(SweepRange, __sweeprange_representer)
    yaml.representer.add_representer(SweepManual, __sweepmanual_representer)
    yaml.representer.add_representer(type(None), __represent_none)
    buffer = StringIO()
    yaml.dump(pipeline_full, buffer)
    atomic_write(Path(path_to_output_file), buffer.getvalue())

    return 0


def _add_comment(mapping, *, key, comment):
    if key in mapping:
        mapping.yaml_add_eol_comment(key=key, comment=comment)


def _set_sweep(parameters, directive, index):
    keys = {key for key in directive if key.startswith("sweep_")}
    if not keys:
        return
    parameter = directive.get("sweep_parameter")
    if not isinstance(parameter, str) or parameter not in parameters:
        raise ValueError(
            f"Step {index}: sweep_parameter must name a template parameter"
        )
    if keys == {"sweep_parameter", "sweep_values"}:
        values = directive["sweep_values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Step {index}: sweep_values must be a nonempty list")
        parameters[parameter] = SweepManual(list(values))
    elif keys == {"sweep_parameter", "sweep_start", "sweep_stop", "sweep_step"}:
        start, stop, step = (
            directive[key] for key in ("sweep_start", "sweep_stop", "sweep_step")
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in (start, stop, step)
        ):
            raise ValueError(f"Step {index}: sweep range values must be finite numbers")
        if step == 0 or (stop - start) * step <= 0:
            raise ValueError(
                f"Step {index}: sweep range must be nonempty with a nonzero step in the correct direction"
            )
        parameters[parameter] = SweepRange(start, stop, step)
    else:
        raise ValueError(
            f"Step {index}: provide either sweep_values or all of sweep_start, sweep_stop and sweep_step"
        )


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that generates YAML pipelines for HTTomo "
        "using YAML templates from httomo-backends."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="A path to the list of pipelines needed to be built within a yaml file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Full path to the yaml file with the generated pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    path_to_httomobackends = os.path.dirname(httomo_backends.__file__)
    args = get_args()
    path_to_pipelines = args.input
    path_to_output_file = args.output
    return_val = yaml_pipelines_generator(
        path_to_pipelines, path_to_httomobackends, path_to_output_file
    )
    if return_val == 0:
        message_str = f"YAML pipeline {path_to_output_file} has been generated."
        print(message_str)
