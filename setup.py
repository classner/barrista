#!/usr/bin/env python
"""Barrista setup script."""
# pylint: disable=C0103
import os
from setuptools import setup

###############################################################################
# install requirements are defined in requirements.txt
# we only want to define those once, so we re-use the file
###############################################################################
# call independent path of the requirements file
file_path_requirements = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')

with open(file_path_requirements, 'r') as fi:
    requirements = fi.read().splitlines()

setup(name='barrista',
      version='0.4',
      description='Serving your caffe right',
      author='Christoph Lassner',
      author_email='classner@tue.mpg.de',
      test_suite='tests',
      packages=['barrista'],
      install_requires=requirements)
