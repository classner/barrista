# -*- coding: utf-8 -*-
"""
Initialization module.

Adds caffe to the pythonpath when imported. Any changes of :mod:`config`
objects must be done before the import. Any imports of caffe related objects
may only be done after.
"""
# pylint: disable=E0611, F0401

from .config import CAFFE_PYTHON_FOLDER as _PYCAFFE_FOLDER
import os as _os
import sys as _sys
import logging as _logging

_LOGGER = _logging.getLogger(__name__)

if not _os.path.exists(_os.path.join(_PYCAFFE_FOLDER, 'caffe')):
    _LOGGER.warn('The caffe module does not exist in %s! It is specified as ' +
                 'barrista.CAFFE_PYTHON_FOLDER! Trying to fall back on ' +
                 'caffe on the python path.',
                 _PYCAFFE_FOLDER)
    try:
        # pylint: disable=W0611
        import caffe as _caffe
    except ImportError:
        _LOGGER.fatal('Failed to add the CAFFE_PYTHON_FOLDER and caffe is ' +
                      'not on the PYTHONPATH!')
        _sys.exit(1)
else:
    _sys.path.insert(0, _PYCAFFE_FOLDER)


def init():
    """Empty at the moment."""
    pass
