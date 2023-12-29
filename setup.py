from setuptools import setup, Extension
import numpy

activations_module = Extension(
    "inference.activations",
    sources=["src/activations.c"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="numpy-forward",
    version="0.1",
    description="Python package for inference using numpy",
    packages=["src/inference"],
    ext_modules=[activations_module],
)
