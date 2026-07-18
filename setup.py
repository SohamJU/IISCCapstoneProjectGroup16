"""Package setup file."""

from setuptools import find_packages, setup


setup(
    packages=find_packages(where="."),
    include_package_data=True,
)
