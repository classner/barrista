#!/usr/bin/env python
"""Preparing the data."""
# pylint: disable=invalid-name, no-member
from __future__ import print_function

import os as _os
import logging as _logging
import cv2 as _cv2
import numpy as _np

import click as _click
import progressbar as _progressbar
import skdata.cifar10 as _skdc10


_LOGGER = _logging.getLogger(__name__)
_DATA_FOLDER = _os.path.join(_os.path.dirname(__file__),
                             'data')
_MEAN = None
if not _os.path.exists(_DATA_FOLDER):
    _LOGGER.info("Data folder not found. Creating...")
    _os.mkdir(_DATA_FOLDER)


def training_data():
    """Get the `CIFAR-10` training data."""
    global _MEAN  # pylint: disable=global-statement
    _np.random.seed(1)
    view = _skdc10.view.OfficialImageClassificationTask()
    permutation = _np.random.permutation(range(50000))
    if _MEAN is None:
        _MEAN = view.train.x.reshape((50000 * 32 * 32, 3)).mean(axis=0)
    return ((view.train.x[:50000, :][permutation, :] - _MEAN).
            transpose((0, 3, 1, 2)).astype('float32'),
            view.train.y[:50000][permutation].reshape((50000, 1)).astype('float32'))


def test_data():
    """Get the `CIFAR-10` test data."""
    global _MEAN  # pylint: disable=global-statement
    _np.random.seed(1)
    view = _skdc10.view.OfficialImageClassificationTask()
    permutation = _np.random.permutation(range(10000))
    if _MEAN is None:
        _MEAN = view.train.x.reshape((50000 * 32 * 32, 3)).mean(axis=0)
    return ((view.test.x[:10000, :][permutation, :] - _MEAN).
            transpose((0, 3, 1, 2)).astype('float32'),
            view.test.y[:10000][permutation].reshape((10000, 1)).astype('float32'))


@_click.group()
def _cli():
    """Handle the experiment data."""
    pass

@_cli.command()
def validate_storage():
    """Validate the data."""
    _LOGGER.info("Validating storage...")
    val_folder = _os.path.join(_DATA_FOLDER, 'images')
    _LOGGER.info("Writing images to %s.",
                 val_folder)
    if not _os.path.exists(val_folder):
        _os.mkdir(val_folder)
    _LOGGER.info("Train...")
    tr_folder = _os.path.join(val_folder, 'train')
    if not _os.path.exists(tr_folder):
        _os.mkdir(tr_folder)
    tr_data, tr_labels = training_data()
    _LOGGER.info("Mean determined as: %s.", str(_MEAN))
    pbar = _progressbar.ProgressBar(maxval=50000 - 1,
                                    widgets=[_progressbar.Percentage(),
                                             _progressbar.Bar(),
                                             _progressbar.ETA()])
    pbar.start()
    for idx in range(50000):
        _cv2.imwrite(_os.path.join(tr_folder, '%05d_%d.jpg' % (idx,
                                                               int(tr_labels[idx, 0]))),
                     (tr_data[idx,].transpose((1, 2, 0)) + _MEAN).astype('uint8'))
        pbar.update(idx)
    pbar.finish()
    _LOGGER.info("Test...")
    te_folder = _os.path.join(val_folder, 'test')
    if not _os.path.exists(te_folder):
        _os.mkdir(te_folder)
    te_data, te_labels = test_data()
    pbar = _progressbar.ProgressBar(maxval=10000 - 1,
                                    widgets=[_progressbar.Percentage(),
                                             _progressbar.Bar(),
                                             _progressbar.ETA()])
    pbar.start()
    for idx in range(10000):
        _cv2.imwrite(_os.path.join(te_folder, '%05d_%d.jpg' % (idx,
                                                               int(te_labels[idx, 0]))),
                     (te_data[idx,].transpose((1, 2, 0)) + _MEAN).astype('uint8'))
        pbar.update(idx)
    pbar.finish()

if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    _cli()
