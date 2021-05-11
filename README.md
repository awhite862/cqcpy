# cqcpy: Some quantum chemistry utilities in python
These utilies exist to provide some generally useful functionality for development
and testing of quantum chemistry code.

[![Tests](https://github.com/awhite862/cqcpy/workflows/Tests/badge.svg)](https://github.com/awhite862/cqcpy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/awhite862/cqcpy/branch/master/graph/badge.svg)](https://codecov.io/gh/awhite862/cqcpy)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/awhite862/cqcpy.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/awhite862/cqcpy/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/awhite862/cqcpy.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/awhite862/cqcpy/context:python)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/awhite862/cqcpy/master/LICENSE)

## Prerequisites
+ numpy
+ pyscf (optional)

## Features
+ coupled cluster energy and amplitude equations
+ functions/classes for storing and manipulating determinant strings
+ compute Fermi-Dirac occupations and related quantities
+ convenience functions for getting integrals from pyscf in various formats
+ classes/functions for storing and transforming some commonly found tensor structures
+ functions to generate random tensors useful for various tests

## Caveats
+ These routines are not designed for efficiency
+ Test coverage varies, so use at your own risk!

## Tests
Tests can be run in any of the following ways:
  - Individually from the `cqcpy/tests` subdirectory
  - All at once by running `python test_suites.py` from `cqcpy/tests`
  - All at once by running `python test.py`
