HTTomo-backends
---------------

Purpose of HTTomo-backends
==========================

**HTTomo-backends** is a package to support `HTTomo <https://diamondlightsource.github.io/httomo/>`_ software and it contains the following elements:

* YAML templates for `backends <https://diamondlightsource.github.io/httomo/backends/list.html>`_. Those include methods from those libraries. The YAML templates can be accessed directly in the environment where `httomo-backends` is installed. 
* Scripts that automatically generate YAML templates during documentation build.
* Supporting scripts to calculate memory, size and padding estimators of the methods


Installation
============

HTTomo-backends is available on PyPI, but currently can only be installed into a conda
environment (due to a dependency being available only through conda).

.. code-block:: console

   $ conda create --name httomo-backends
   $ conda activate httomo-backends
   $ conda install -c ccpi -c conda-forge ccpi-regulariser cupy==12.3.0
   $ pip install httomo-backends
