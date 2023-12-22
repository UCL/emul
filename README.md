# emul

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Licence][licence-badge]](./LICENCE.md)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/UCL/emul/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/UCL/emul/actions/workflows/tests.yml
[linting-badge]:            https://github.com/UCL/emul/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/UCL/emul/actions/workflows/linting.yml
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/emul
[conda-link]:               https://github.com/conda-forge/emul-feedstock
[pypi-link]:                https://pypi.org/project/emul/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/emul
[pypi-version]:             https://img.shields.io/pypi/v/emul
[licence-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

Python implementations of Gaussian process emulators.

This project is developed in collaboration with the [Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.

## About

### Project Team

- Matt Graham ([matt-graham](https://github.com/matt-graham))

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`emul` requires Python 3.10&ndash;3.12.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using a environment management tool such as [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) or [Conda](https://conda.io/projects/conda/en/latest/). To install the latest development version of `emul` using `pip` in the currently active environment run

```sh
pip install git+https://github.com/UCL/emul.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/UCL/emul.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running locally

How to run the application on your local system.

### Running tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments using
[`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

## Acknowledgements

This work was funded by a grant from the ExCALIBUR programme.
