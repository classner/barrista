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
from sklearn.datasets import fetch_mldata as _fetch_mldata


_LOGGER = _logging.getLogger(__name__)
_DATA_FOLDER = _os.path.join(_os.path.dirname(__file__),
                             'data')
if not _os.path.exists(_DATA_FOLDER):
    _LOGGER.info("Data folder not found. Creating...")
    _os.mkdir(_DATA_FOLDER)


def training_data():
    """Get the `MNIST original` training data."""
    _np.random.seed(1)
    permutation = _np.random.permutation(range(60000))
    mnist = _fetch_mldata('MNIST original',
                          data_home=_os.path.join(_DATA_FOLDER,
                                                  'MNIST_original'))
    return (mnist.data[:60000, :][permutation, :].reshape((60000, 1, 28, 28)).astype('float32'),
            mnist.target[:60000][permutation].reshape((60000, 1)).astype('float32'))


def test_data():
    """Get the `MNIST original` test data."""
    mnist = _fetch_mldata('MNIST original',
                          data_home=_os.path.join(_DATA_FOLDER,
                                                  'MNIST_original'))
    return (mnist.data[60000:, :].reshape((10000, 1, 28, 28)).astype('float32'),
            mnist.target[60000:].reshape((10000, 1)).astype('float32'))


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
    pbar = _progressbar.ProgressBar(maxval=60000 - 1,
                                    widgets=[_progressbar.Percentage(),
                                             _progressbar.Bar(),
                                             _progressbar.ETA()])
    pbar.start()
    for idx in range(60000):
        _cv2.imwrite(_os.path.join(tr_folder, '%05d_%d.jpg' % (idx,
                                                               int(tr_labels[idx, 0]))),
                     tr_data[idx, 0])
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
                     te_data[idx, 0])
        pbar.update(idx)
    pbar.finish()

if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    _cli()
