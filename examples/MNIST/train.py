#!/usr/bin/env python
"""Training of the network."""
# pylint: disable=wrong-import-position
from __future__ import print_function
import os
import sys
import imp
import shutil
import glob
import logging

import click
import barrista.solver as sv
import barrista.net as bnet
import barrista.monitoring as mnt

from data import training_data, test_data


_LOGGER = logging.getLogger(__name__)
LOGFORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
RESULT_FOLDER = os.path.join(os.path.dirname(__file__),
                             'results')
if not os.path.exists(RESULT_FOLDER):
    os.mkdir(RESULT_FOLDER)


# pylint: disable=too-many-arguments, too-many-locals
def _model(result_folder,
           epoch_size,
           model_name=None,
           epoch=None,
           optimizer_name='adam',
           lr_param=0.01,
           mom_param=0.9,
           wd_param=1E-4,
           no_solver=False,
           allow_overwrite=False):
    """Get a model and optimizer either loaded or created."""
    optimizer_name = str(optimizer_name)
    out_folder = os.path.join('results', result_folder)
    if optimizer_name == 'sgd':
        optimizer = sv.SGDSolver(base_lr=lr_param,
                                 momentum=mom_param,
                                 weight_decay=wd_param,
                                 snapshot_prefix=os.path.join(
                                     str(out_folder), 'model'))
    else:
        optimizer = sv.AdamSolver(base_lr=lr_param,  # pylint: disable=redefined-variable-type
                                  weight_decay=wd_param,
                                  snapshot_prefix=os.path.join(
                                      str(out_folder), 'model'))
    if os.path.exists(os.path.join('results', result_folder)) and (
            not allow_overwrite or (allow_overwrite and model_name is None)):
        assert model_name is None, (
            "This result path already exists! "
            "If you still want to use it, add the flag `--allow_overwrite`.")
        logging.basicConfig(
            level=logging.DEBUG,
            format=LOGFORMAT,
            filename=os.path.join('results', result_folder, 'train.log'),
            filemode='a')
        _LOGGER.info("Provided arguments: %s.", str(sys.argv))
        # Load the data from there.
        modelmod = imp.load_source(
            '_modelmod',
            os.path.join('results', result_folder, 'model.py'))
        model = modelmod.MODEL
        checkpoint_step = epoch_size / model.blobs['data'].shape[0]
        if epoch is None:
            # Use the last one.
            epoch = 1
            while os.path.exists(os.path.join('results',
                                              result_folder,
                                              'model_iter_%d.caffemodel' % (
                                                  epoch * checkpoint_step))):
                epoch += 1
            epoch -= 1
        cmfilename = os.path.join('results',
                                  result_folder,
                                  'model_iter_%d.caffemodel' % (
                                      epoch * checkpoint_step))
        ssfilename = os.path.join('results',
                                  result_folder,
                                  'model_iter_%d.solverstate' % (
                                      epoch * checkpoint_step))

        assert os.path.exists(cmfilename), (
            "Could not find model parameter file at %s!" % (cmfilename))
        assert os.path.exists(ssfilename), (
            "Could not find solverstate file at %s!" % (ssfilename))

        _LOGGER.info("Loading model from %s...", cmfilename)
        model.load_blobs_from(str(cmfilename))
        if not no_solver:
            _LOGGER.info("Loading solverstate from %s...", ssfilename)
            optimizer.restore(str(ssfilename), model)
        base_iter = epoch * epoch_size
    else:
        # Create the result folder.
        assert model_name is not None, (
            "If a new result_folder is specified, a model name must be given!")
        out_folder = os.path.join(RESULT_FOLDER, result_folder)
        if os.path.exists(out_folder):
            # Reset, because an overwrite was requested.
            shutil.rmtree(out_folder)
        os.mkdir(out_folder)
        os.mkdir(os.path.join(out_folder, 'visualizations'))
        logging.basicConfig(
            level=logging.DEBUG,
            format=LOGFORMAT,
            filename=os.path.join(out_folder, 'train.log'),
            filemode='w')
        _LOGGER.info("Provided arguments: %s.", str(sys.argv))
        _LOGGER.info("Result folder created: %s.", out_folder)
        _LOGGER.info("Freezing experimental setup...")
        # Copy the contents over.
        shutil.copy2(os.path.join('models', model_name + '.py'),
                     os.path.join(out_folder, 'model.py'))
        for pyfile in glob.glob(os.path.join(os.path.dirname(__file__),
                                             '*.py')):
            shutil.copy2(pyfile,
                         os.path.join(out_folder, os.path.basename(pyfile)))
        _LOGGER.info("Creating model...")
        # Get the model.
        modelmod = imp.load_source('_modelmod',
                                   os.path.join(out_folder, 'model.py'))
        model = modelmod.MODEL
        base_iter = 0
    if no_solver:
        return model, None, out_folder, base_iter
    else:
        return model, optimizer, out_folder, base_iter


@click.command()
@click.argument("result_folder", type=click.STRING)  # pylint: disable=no-member
@click.option("--model_name", type=click.STRING,
              help='Model name to use, if a new trial should be created.')
@click.option("--epoch", type=click.INT, default=None,
              help='Epoch to start from, if training is resumed.')
@click.option("--num_epoch", type=click.INT, default=3,
              help='Final number of epochs to reach. Default: 3.')
@click.option("--optimizer_name", type=click.Choice(['adam', 'sgd']), default='adam',
              help='Optimizer to use. Default: adam.')
@click.option("--lr_param", type=click.FLOAT, default=0.001,
              help='The base learning rate to use. Default: 0.001.')
@click.option("--mom_param", type=click.FLOAT, default=0.9,
              help='The momentum to use if SGD is the optimizer. Default: 0.9.')
@click.option("--wd_param", type=click.FLOAT, default=0.0001,
              help='The weight decay to use. Default: 0.0001.')
@click.option("--allow_overwrite", type=click.BOOL, default=False, is_flag=True,
              help='Allow reuse of an existing result directory.')
@click.option("--use_cpu", type=click.BOOL, default=False, is_flag=True,
              help='Use the CPU. If not set, use the GPU.')
# pylint: disable=too-many-arguments, unused-argument
def cli(result_folder,
        model_name=None,
        epoch=None,
        num_epoch=3,
        optimizer_name='adam',
        lr_param=0.001,
        mom_param=0.9,
        wd_param=0.0001,
        allow_overwrite=False,
        use_cpu=False):
    """Train a model."""
    print("Parameters: ", sys.argv)
    if use_cpu:
        bnet.set_mode_cpu()
    else:
        bnet.set_mode_gpu()
    # Load the data.
    tr_data, tr_labels = training_data()
    te_data, te_labels = test_data()
    # Setup the output folder, including logging.
    model, optimizer, out_folder, base_iter = _model(
        result_folder,
        tr_data.shape[0],
        model_name,
        epoch,
        optimizer_name,
        lr_param,
        mom_param,
        wd_param,
        False,
        allow_overwrite)
    logger = mnt.JSONLogger(str(out_folder),
                            'model',
                            {'train': ['train_loss', 'train_accuracy'],
                             'test': ['test_loss', 'test_accuracy']},
                            base_iter=base_iter,
                            write_every=10000,
                            create_plot=True)
    progr_ind = mnt.ProgressIndicator()

    model.fit(num_epoch * tr_data.shape[0],
              optimizer,
              X={'data': tr_data, 'labels': tr_labels},
              X_val={'data': te_data, 'labels': te_labels},
              test_interval=tr_data.shape[0],
              train_callbacks=[
                  progr_ind,
                  logger,
                  mnt.GradientMonitor(10000,
                                      os.path.join(str(out_folder),
                                                   'visualizations' + os.sep),
                                      relative=False),
                  mnt.GradientMonitor(10000,
                                      os.path.join(str(out_folder),
                                                   'visualizations' + os.sep),
                                      relative=True),
                  mnt.ActivationMonitor(10000,
                                        os.path.join(str(out_folder),
                                                     'visualizations' + os.sep),
                                        sample={'data': tr_data[0]}),
                  mnt.FilterMonitor(10000,
                                    os.path.join(str(out_folder),
                                                 'visualizations' + os.sep)),
                  mnt.Checkpointer(os.path.join(str(out_folder),
                                                'model'),
                                   tr_data.shape[0])],
              test_callbacks=[
                  progr_ind,
                  logger])


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
