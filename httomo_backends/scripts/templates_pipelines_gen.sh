#!/usr/bin/env bash

python /home/algol/Documents/DEV/httomo-backends/httomo_backends/scripts/yaml_templates_generator.py -i /home/algol/Documents/DEV/httomo-backends/httomo_backends/methods_database/packages/backends/tomopy/tomopy_modules.yaml -o /home/algol/miniforge3/envs/httomo_cupy14/lib/python3.12/site-packages/httomo_backends/yaml_templates/tomopy
python /home/algol/Documents/DEV/httomo-backends/httomo_backends/scripts/yaml_templates_generator.py -i /home/algol/Documents/DEV/httomo-backends/httomo_backends/methods_database/packages/backends/httomolibgpu/httomolibgpu_modules.yaml -o /home/algol/miniforge3/envs/httomo_cupy14/lib/python3.12/site-packages/httomo_backends/yaml_templates/httomolibgpu
python /home/algol/Documents/DEV/httomo-backends/httomo_backends/scripts/yaml_templates_generator.py -i /home/algol/Documents/DEV/httomo-backends/httomo_backends/methods_database/packages/backends/httomolib/httomolib_modules.yaml -o /home/algol/miniforge3/envs/httomo_cupy14/lib/python3.12/site-packages/httomo_backends/yaml_templates/httomolib


python /home/algol/Documents/DEV/httomo/docs/source/scripts/execute_pipelines_build.py -o /home/algol/Documents/DEV/httomo/docs/source/pipelines_full/
