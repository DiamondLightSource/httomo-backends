HTTomo-backends release how-to
------------------------------

In order to update the yaml templates on the documentation page and build the `httomo-backends` PyPi package (with the updated templates) one need to do the steps bellow: 

* update version of `httomo-backends`` in the `pyproject.toml` file. Change it respectively to changes in the libraries as well as httomo. For example, if it is a simple update of parameters of the existing methods in libraries change the micro version, if a new method added to libraries change the minor version and if there were significant, possibly backward incompatible, changes in httomo and supporting scripts, then change the major version. 

* If one needs to update TomoPy's templates then update the version of `tomopy` in `./docs/source/doc-conda-requirements.yml`

* The same applies for the PyPi build, update the `tomopy` version in `/.github/workflows/httomo_backends_pypi_publish.yml`,  if needed. 

* Tag the `main` branch and make a release to trigger the PyPi build. 

Please note that there might be no need to update `httomolib` and `httomolibgpu` versions as by default the pip installed versions will be the latest.



