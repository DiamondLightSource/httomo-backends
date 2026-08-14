Properties and execution requirements
=====================================

In order for HTTomo to execute a method it requires certain information about the method, such
as its pattern, and (if it's a GPU method) the amount of GPU memory required per-slice.

Some of this information is simple (such as the pattern of a method), and other information may
be more complex and thus could require a function to describe it (such as the GPU memory usage
of a GPU method).

The simple information describes properties of a method (its pattern, whether it's a CPU or GPU
method, etc), and the complex information describes requirements for executing the method (how
much GPU memory is needed per-slice, how much padding is needed per-slice, etc).

In :code:`httomo-backends`, the properties of methods are stored in YAML files (which are often
referred to as "library files"), and the execution requirements are stored in python functions
(which are often referred to as "supporting functions").

Library files
-------------

Each supported backend has modules which contain methods, and the library files are organised
similarly:

- each backend has a library file
- a library file has a section for each of the modules within the associated backend
- for a given method, the simple information for that method is stored in the library file for
  the backend the method is in

For example, :code:`httomolibgpu` has a method :code:`normalize`. The simple information for
this :code:`normalize` method is stored in the libary file for :code:`httomolibgpu`.

Backend library files
---------------------

Below is a list of library files for the currently supported backends.

.. dropdown:: TomoPy

    .. literalinclude:: ../../../httomo_backends/methods_database/packages/backends/tomopy/tomopy.yaml

.. dropdown:: Httomolibgpu

    .. literalinclude:: ../../../httomo_backends/methods_database/packages/backends/httomolibgpu/httomolibgpu.yaml

.. dropdown:: Httomolib

    .. literalinclude:: ../../../httomo_backends/methods_database/packages/backends/httomolib/httomolib.yaml
