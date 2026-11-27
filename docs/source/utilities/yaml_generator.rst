.. _utilities_yamlgenerator:

Templates generator
*******************
:ref:`backends_content` can be generated automatically using the YAML generator tool `provided <https://github.com/DiamondLightSource/httomo-backends/blob/main/httomo_backends/scripts/yaml_templates_generator.py>`_ in the Github repo.

The script does the following:

* Generates a list of YAML files for all accessible on import methods in a chosen software package, e.g., TomoPy.
* Modifies and/or removes some extracted parameters in YAML templates to make the templates compatible with HTTomo.

How does it work:

* The user would need to provide a YAML file with the listed *modules* you would like to inspect and extract the methods from. For instance, for the TomoPy package this would be:

.. code-block:: yaml

    - tomopy.misc.corr
    - tomopy.misc.morph

* The generator can be applied using the following command:

.. code-block:: console

   $ python -m httomo_backends.scripts.yaml_templates_generator -i /path/to/modules.yaml -o /path/to/outputfolder/

Please note that the package from which the modules are extracted, must be installed into your conda environment.

**For TomoPy templates only.** After templates have been generated for TomoPy, we need to remove the ones that are not currently supported by HTTomo. We do that by looking into the library file that exists in HTTomo for TomoPy.

.. code-block:: console

   $ python -m remove_unsupported_templates -t /path/to/templates/ -l /path/to/library/file

Keeping templates compatible
----------------------------

For registered HTTomolib and HTTomolibGPU methods, generate templates from the
versions installed in the current environment. Run from the repository root::

    python -m httomo_backends.scripts.yaml_templates_generator --supported-only \
        -i httomo_backends/methods_database/packages/backends/httomolib/httomolib_modules.yaml \
        -o httomo_backends/yaml_templates/httomolib
    python -m httomo_backends.scripts.yaml_templates_generator --supported-only \
        -i httomo_backends/methods_database/packages/backends/httomolibgpu/httomolibgpu_modules.yaml \
        -o httomo_backends/yaml_templates/httomolibgpu

``--supported-only`` excludes methods without a database registration. The
parameter defaults follow each API, with HTTomo conventions for automatic axes,
centre-finding side outputs and asynchronous image saving. Runtime-supplied data,
calibration frames and internal memory-estimation controls are omitted.

Templates are starting points. Replace ``REQUIRED`` values and experiment-specific
parameters before running a pipeline. For distortion correction, provide either
a calibration ``metadata_path`` or ``xcenter``, ``ycenter`` and ``list_fact``.
Paganin pixel size is in microns, distance in metres and energy in keV; use
``ratio_delta_beta`` to set material contrast.

Check API compatibility
-----------------------

Install the library versions you intend to use, including ToMoBAR for the GPU
reconstruction wrappers, and run::

    python -m httomo_backends.scripts.validate_api
    python -m pytest tests/test_api_compatibility.py tests/test_method_query.py

The command requires both libraries. Use ``--package httomolib`` or
``--package httomolibgpu`` to inspect one library. It checks registrations,
module lists, template parameters and defaults, and pipeline directive parameter
names, including sweep parameters. It discovers newly exported processing methods
and reports methods without a registration or an explicit deferral. The pytest
checks skip a library only when it is not installed; the command fails if a
requested library is missing. Neither check runs reconstruction or GPU kernels.

Split Bregman TV, Fast Gradient Projection TV, Total Generalised Variation and
nonlinear diffusion are available as ``httomolibgpu.misc.denoise.total_variation_SB``,
``httomolibgpu.misc.denoise.total_variation_FGP``,
``httomolibgpu.misc.denoise.total_generalised_variation`` and
``httomolibgpu.misc.denoise.nonlinear_diffusion``.
Their module memory estimates account for a possible non-C-contiguous input
copy and, for the TV methods, half precision. Nonlinear diffusion requires
``half_precision: false``. They follow the existing TV five-slice overlap
policy for slab processing. This overlap is an approximation: it does not
guarantee the same result as processing the whole volume at once.

The remaining public GPU method is deferred until its execution requirements
are supported:

* ``iterative_alignment`` needs processing and memory support, plus handling of
  the reconstructed volume, detector shifts and optional diagnostics.

These deferrals are recorded in ``httomo_backends.scripts.validate_api``. Remove
an entry when its method is registered; the checker reports stale deferrals.
Numerical correctness, memory estimates and pipeline execution require separate
validation.

Building a pipeline from directives
-----------------------------------

Expand a directive file into a pipeline with an explicit output file::

    python -m httomo_backends.scripts.yaml_pipelines_generator \
        -i directives.yaml -o output/pipeline.yaml

Each directive requires ``module_path`` and ``method``. Optional ``parameters``
override the corresponding template values; parameter names must appear in that
template. ``id``, ``side_outputs`` and ``save_result`` can also override template
metadata. For example:

.. code-block:: yaml

    - module_path: httomolib.prep.phase
      method: paganin_filter
      parameters:
        pixel_size: 1.28
        distance: 1.0
        ratio_delta_beta: 150

A sweep requires ``sweep_parameter`` and either a nonempty ``sweep_values`` list
or all three of ``sweep_start``, ``sweep_stop`` and ``sweep_step``. Range stops
are exclusive. Steps must be nonzero and point towards the stop value.
The named parameter must exist in the template. Each sweep applies only to its
own method.

Both generators validate and serialise their inputs before replacing existing
outputs. Invalid directives, missing templates or unsupported parameter defaults
therefore leave existing output files intact. Required command-line paths and
invalid input structures are reported as errors.
