# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:07:56 2018

@author: Walton TWang

Setup script for slim."""

from setuptools import find_packages
from setuptools import setup


setup(
    name='slim',
    version='0.1',
    include_package_data=True,
    packages=find_packages(),
    description='tf-slim',
)
