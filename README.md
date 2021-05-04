# cqcpy: Some quantum chemistry utilities in python
These utilies exist to provide some generally useful functionality for development
and testing of quantum chemistry code.

[![Build](https://github.com/awhite862/cqcpy/workflows/Build/badge.svg)](https://github.com/awhite862/cqcpy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/awhite862/cqcpy/branch/master/graph/badge.svg)](https://codecov.io/gh/awhite862/cqcpy)

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
