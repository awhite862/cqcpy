"""Setup script for cqcpy
"""
import os
from setuptools import setup, find_packages, Extension


def get_version(fname):
    gvar = {}
    with open(fname) as f:
        exec(f.read(), gvar)
    return gvar["__version__"]


def run():
    """Run the setup script
    """
    cdir = os.path.dirname(os.path.abspath(__file__))
    version_path = os.path.join(cdir, "cqcpy", "version.py")
    __version__ = get_version(version_path)
    setup(name='cqcpy',
          version=__version__,
          author="Alec White",
          description="Some quantum chemistry utilities",
          install_requires=[
              "numpy",
              "pyscf"],
          license="MIT",
          packages=["cqcpy"])


if __name__ == "__main__":
    run()
