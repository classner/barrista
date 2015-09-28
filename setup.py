#!/usr/bin/env python
from setuptools import setup

setup(name='barrista',
      version='0.2',
      description='Serving your caffe right',
      author='Christoph Lassner',
      author_email='classner@tue.mpg.de',
      test_suite='tests',
      packages=['barrista'],
      install_requires=[
          'numpy',
          'protobuf',
          'progressbar',
          'scikit-learn',
          'scikit-image',
          'scipy'
      ])
