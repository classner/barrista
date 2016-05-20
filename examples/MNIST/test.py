#!/usr/bin/env python
"""Create visualizations."""
# pylint: disable=no-member, invalid-name
from __future__ import print_function

import logging
import click
import numpy as np
from sklearn.metrics import accuracy_score

import barrista.monitoring as mnt
import barrista.net as bnet
from train import _model
from data import training_data, test_data

_LOGGER = logging.getLogger(__name__)


@click.group()
def cli():
    """Test a model."""
    pass


@cli.command()
@click.argument('result_folder', type=click.STRING)
@click.option('--epoch', type=click.INT, default=None,
              help="The epoch of the model to use.")
@click.option('--image_idx', type=click.INT, default=0,
              help="The image to visualize.")
@click.option("--use_cpu", type=click.BOOL, default=False, is_flag=True,
              help='Use the CPU. If not set, use the GPU.')
# pylint: disable=too-many-locals
def test_image(
        result_folder,
        epoch=None,
        image_idx=0,
        use_cpu=False):
    """Test a network on one test image."""
    if use_cpu:
        bnet.set_mode_cpu()
    else:
        bnet.set_mode_gpu()
    _LOGGER.info("Loading data...")
    tr_data, _ = training_data()
    te_data, _ = test_data()
    _LOGGER.info("Loading network...")
    # Load the model for training.
    model, _, _, _ = _model(result_folder,
                            tr_data.shape[0],
                            epoch=epoch)
    _LOGGER.info("Predicting...")
    results = model.predict(te_data,
                            test_callbacks=[mnt.ProgressIndicator()])
    _LOGGER.info("Prediction for image %d: %s.",
                 image_idx, str(results[image_idx]))


@cli.command()
@click.argument('result_folder', type=click.STRING)
@click.option('--epoch', type=click.INT, default=None,
              help="The epoch of the model to use.")
@click.option("--use_cpu", type=click.BOOL, default=False, is_flag=True,
              help='Use the CPU. If not set, use the GPU.')
# pylint: disable=too-many-locals
def score(
        result_folder,
        epoch=None,
        use_cpu=False):
    """Test a network on the dataset."""
    if use_cpu:
        bnet.set_mode_cpu()
    else:
        bnet.set_mode_gpu()
    _LOGGER.info("Loading data...")
    tr_data, _ = training_data()
    te_data, te_labels = test_data()
    _LOGGER.info("Loading network...")
    # Load the model.
    model, _, _, _ = _model(result_folder,
                            tr_data.shape[0],
                            epoch=epoch,
                            no_solver=True)
    _LOGGER.info("Predicting...")
    results = model.predict(te_data,
                            test_callbacks=[mnt.ProgressIndicator()])
    _LOGGER.info("Accuracy: %f.",
                 accuracy_score(te_labels,
                                np.argmax(np.array(results), axis=1)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()
