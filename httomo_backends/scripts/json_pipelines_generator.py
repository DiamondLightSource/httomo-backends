#!/usr/bin/env python3
import importlib
import inspect
import yaml
from typing import Dict, Any
import os

"""This script provides a function `json_pipeline_generator` to generate JSON objects 
containing method and module names out of directive YAML files.

to generate JSON for single directive yaml file : 
from generate_pipeline_json import process_all_yaml_files

result = json_pipeline_generator(<address to yaml_file>)

to generate a JSON for all directive yaml files in the pipelines_full directory :
from generate_pipeline_json import process_all_yaml_files

result = process_all_yaml_files()
"""

# Directory containing YAML files relative to this script
PIPELINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines_full")

ignored_methods = ["standard_tomo","calculate_stats"]

def get_yaml_path(yaml_filename: str) -> str:
    """
    Get the full path to a YAML file in the pipelines_full directory.
    
    Args:
        yaml_filename: Name of the YAML file (e.g., "example.yaml")
    
    Returns:
        Full path to the YAML file
    """
    return os.path.join(PIPELINES_DIR, yaml_filename)

def import_module_safely(module_name: str):
    """
    Safely import a module handling potential import errors
    
    Args:
        module_name: The name of the module to import
    
    Returns:
        The imported module or None if import fails
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
        print(f"Warning: Could not import dependency '{missing_module}' for module '{module_name}'")
        print(f"You may need to install it using: pip install {missing_module}")
        return None
    except Exception as e:
        print(f"Warning: Error importing module '{module_name}': {str(e)}")
        return None

def inspect_method_parameters(module_name: str, method_name: str) -> Dict[str, Any]:
    """
    Inspect a method's parameters from a given module
    
    Args:
        module_name: The full path to the module
        method_name: Name of the method to inspect
    
    Returns:
        Dictionary of parameter names and their default values
    """
    # Import the module safely
    module = import_module_safely(module_name)
    if module is None:
        # Return empty parameters if module import failed
        return {}    
    try:
        # Get the method
        method = getattr(module, method_name)
        
        # Get method signature
        signature = inspect.signature(method)
        
        # Create parameters dictionary
        parameters = {}
        
        # List of parameters to skip
        skip_params = [
            "in_file", "data_in", "tomo", "arr", "prj", "data",
            "ncore", "nchunk", "flats", "flat", "dark", "darks",
            "theta", "out", "ang", "comm_rank", "out_dir", "angles",
            "gpu_id", "comm", "offset", "shift_xy", "step_xy", 
            "jpeg_quality", "watermark_vals"
        ]
        
        # Process parameters
        for param_name, param in signature.parameters.items():
            if param_name not in skip_params:
                if param_name in ["proj1", "proj2", "axis"]:
                    parameters[param_name] = "auto"
                elif param_name == "asynchronous":
                    parameters[param_name] = True
                elif param_name == "center":
                    parameters[param_name] = "${centering.side_outputs.centre_of_rotation}"
                elif param_name == "glob_stats":
                    parameters[param_name] = "${statistics.side_outputs.glob_stats}"
                elif param_name == "overlap":
                    parameters[param_name] = "${centering.side_outputs.overlap}"
                else:
                    # Get default value if it exists, otherwise mark as REQUIRED
                    default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
                    parameters[param_name] = default
        
        return parameters
        
    except Exception as e:
        print(f"Warning: Error inspecting method '{method_name}' in module '{module_name}': {str(e)}")
        return {}

def json_pipeline_generator(input_yaml: str) -> Dict[str, Any]:
    """
    Generate JSON pipeline from YAML directive
    
    Args:
        input_yaml: Path to input YAML directive file
    
    Returns:
        Dictionary with methods information if successful, or empty dictionary if error occurred
    """
    try:
        # Read input YAML
        with open(input_yaml, 'r') as file:
            pipeline = yaml.safe_load(file)
        
        # Dictionary to store methods information
        methods_info = {}
        
        # Process each method in pipeline
        for item in pipeline:
            method_name = item['method']
            module_path = item['module_path']
            # Skip ignored methods
            if method_name not in ignored_methods:  
                print(f"Processing method: {method_name} from module: {module_path}")
                # Get method parameters
                parameters = inspect_method_parameters(module_path, method_name)
                # Add to methods info even if parameters are empty
                methods_info[method_name] = {
                    'module_path': module_path,
                    'parameters': parameters
                }
        return methods_info
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {}

import json
import os

# Path to the priority configuration file
PRIORITY_FILE = os.path.join(PIPELINES_DIR, "pipeline_priority.json")

def load_priority_order() -> list:
    """
    Load the pipeline priority order from the JSON configuration file in the pipelines_full directory.
    
    Returns:
        List of pipeline titles in priority order.
    """
    try:
        with open(PRIORITY_FILE, 'r') as file:
            config = json.load(file)
            return config.get("pipeline_order", [])
    except FileNotFoundError:
        print(f"Warning: Priority file '{PRIORITY_FILE}' not found. Using default order.")
        return []
    except Exception as e:
        print(f"Warning: Could not load priority file '{PRIORITY_FILE}'. Using default order. Error: {str(e)}")
        return []

def process_all_yaml_files() -> Dict[str, Any]:
    """
    Process all YAML files in the pipelines_full directory in the order of priority.
    
    Returns:
        Dictionary where keys are YAML file names and values are the JSON outputs
        from the json_pipeline_generator function.
    """
    # Load the priority order
    priority_order = load_priority_order()
    
    # Dictionary to store results
    results = {}
    
    # List all YAML files in the pipelines_full directory
    yaml_files = [f for f in os.listdir(PIPELINES_DIR) if f.endswith(".yaml") or f.endswith(".yml")]
    
     # Exclude the priority file from processing
    yaml_files = [f for f in yaml_files if f != "pipeline_priority.yaml"]

    # Sort YAML files based on priority order
    if priority_order:
        # Remove the "_directive.yaml" suffix for matching
        yaml_files.sort(key=lambda x: priority_order.index(x.removesuffix("_directive.yaml")) 
                        if x.removesuffix("_directive.yaml") in priority_order 
                        else len(priority_order))
    
    # Process the YAML files in the sorted order
    for yaml_file in yaml_files:
        # Get the full path to the YAML file
        yaml_path = get_yaml_path(yaml_file)
        
        # Process the YAML file
        json_output = json_pipeline_generator(yaml_path)
        
        # Add to results (remove the "_directive.yaml" suffix for the key)
        results[yaml_file.removesuffix("_directive.yaml")] = json_output
    
    return results