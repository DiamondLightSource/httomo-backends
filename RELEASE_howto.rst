HTTomo-backends release how-to
------------------------------

In order to update templates on the documentation page and build the `httomo-backends` PyPi package (with the updated templates) one need to do the following: 

* update version of `httomo-backends`` in the `pyproject.toml` file

* If one needs to update TomoPy's templates then update the version of `tomopy` in `./docs/source/doc-conda-requirements.yml`

* The same applies for the PyPi build, update `tomopy` version there if needed. 


Note that there is usually no need to update `httomolib` and `httomolibgpu` versions anywhere as by default pip installed versions will be the latest.



