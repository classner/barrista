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
import numpy as np
from scipy.misc import imresize

import click
from natsort import natsorted
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


# pylint: disable=too-many-arguments, too-many-locals, too-many-branches
# pylint: disable=too-many-statements
def _model(result_folder,
           epoch_size,
           model_name=None,
           epoch=None,
           write_every=10,
           optimizer_name='sgd',
           lr_param=0.01,
           lr_decay_sched=None,
           lr_decay_ratio=0.1,
           mom_param=0.9,
           wd_param=1E-4,
           no_solver=False,
           allow_overwrite=False):
    """Get a model and optimizer either loaded or created."""
    if epoch is not None:
        write_every = min(write_every, epoch)
    optimizer_name = str(optimizer_name)
    out_folder = os.path.join('results', result_folder)
    if optimizer_name == 'sgd':
        if lr_decay_sched is not None and lr_decay_sched != '':
            lr_policy = 'multistep'
            # Each value must be multiplied with the epoch size (possibly
            # rounded). This is done later once the batch size is known.
            lr_decay_sched = [int(val) for val in lr_decay_sched.split(',')]
        else:
            lr_policy = 'fixed'
        optimizer = sv.SGDSolver(base_lr=lr_param,
                                 momentum=mom_param,
                                 weight_decay=wd_param,
                                 lr_policy=lr_policy,
                                 gamma=lr_decay_ratio,
                                 stepvalue=lr_decay_sched,
                                 snapshot_prefix=os.path.join(
                                     str(out_folder), 'model'))
    else:
        assert lr_decay_sched is not None, (
            "LR decay schedule only supported for SGD!")
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
        batch_size = model.blobs['data'].shape[0]
        checkpoint_step = round_to_mbsize(epoch_size * write_every, batch_size) / batch_size
        if epoch is None:
            # Use the last one.
            modelfiles = glob.glob(os.path.join('results',
                                                result_folder,
                                                'model_iter_*.caffemodel'))
            if len(modelfiles) == 0:
                raise Exception("No model found to resume from!")
            lastm = natsorted(modelfiles)[-1]
            batch_iters = int(os.path.basename(lastm).split('.')[0][11:])
            base_iter = batch_iters * batch_size
            cmfilename = lastm
            ssfilename = cmfilename[:-10] + 'solverstate'
        else:
            assert epoch % write_every == 0, (
                "Writing every %d epochs. Please use a multiple of it!")
            cmfilename = os.path.join('results',
                                      result_folder,
                                      'model_iter_%d.caffemodel' % (
                                          epoch / write_every * checkpoint_step))
            ssfilename = os.path.join('results',
                                      result_folder,
                                      'model_iter_%d.solverstate' % (
                                          epoch / write_every * checkpoint_step))
            base_iter = epoch * epoch_size
        assert os.path.exists(cmfilename), (
            "Could not find model parameter file at %s!" % (cmfilename))
        assert os.path.exists(ssfilename), (
            "Could not find solverstate file at %s!" % (ssfilename))

        _LOGGER.info("Loading model from %s...", cmfilename)
        model.load_blobs_from(str(cmfilename))
        if not no_solver:
            _LOGGER.info("Loading solverstate from %s...", ssfilename)
            if lr_decay_sched is not None:
                # pylint: disable=protected-access
                optimizer._parameter_dict['stepvalue'] = [
                    round_to_mbsize(val * epoch_size, batch_size)
                    for val in lr_decay_sched]
            optimizer.restore(str(ssfilename), model)
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
        if not no_solver and lr_decay_sched is not None:
            # pylint: disable=protected-access
            optimizer._parameter_dict['stepvalue'] = [
                round_to_mbsize(val * epoch_size, batch_size)
                for val in lr_decay_sched]
        base_iter = 0
    if no_solver:
        return model, None, out_folder, base_iter
    else:
        return model, optimizer, out_folder, base_iter


class RandCropMonitor(mnt.ParallelMonitor):

    """Creates random crops."""

    def __init__(self, layer_name, mean, stretch_up_to=1.1, scaling_size=40):
        self._layer_name = layer_name
        self._mean = mean
        self._stretch_up_to = stretch_up_to
        self._scaling_size = scaling_size

    def get_parallel_blob_names(self):
        return [self._layer_name]

    def _pre_train_batch(self, kwargs):
        net = kwargs['net']
        for sample_idx in range(len(net.blobs[self._layer_name].data)):
            sample = net.blobs[self._layer_name].data[sample_idx]
            if sample.min() == 0. and sample.max() == 0.:
                raise Exception("invalid data")
            stretch_factor = np.random.uniform(low=1.,
                                               high=self._stretch_up_to,
                                               size=(2,))
            im = (imresize(sample.transpose((1, 2, 0)) + self._mean,  # pylint: disable=invalid-name
                           (int(self._scaling_size * stretch_factor[0]),
                            int(self._scaling_size * stretch_factor[1])),
                           'bilinear') - self._mean).transpose((2, 0, 1))
            retimx = np.random.randint(low=0,
                                       high=(im.shape[2] -
                                             net.blobs[self._layer_name].data.shape[3] + 1))
            retimy = np.random.randint(low=0,
                                       high=(im.shape[1] -
                                             net.blobs[self._layer_name].data.shape[2] + 1))
            retim = im[:,
                       retimy:retimy+net.blobs[self._layer_name].data.shape[2],
                       retimx:retimx+net.blobs[self._layer_name].data.shape[3]]
            sample[...] = retim

    def _pre_test_batch(self, kwargs):
        net = kwargs['testnet']
        for sample_idx in range(len(net.blobs[self._layer_name].data)):
            sample = net.blobs[self._layer_name].data[sample_idx]
            stretch_factor = (1., 1.)
            im = (imresize(sample.transpose((1, 2, 0)) + self._mean,  # pylint: disable=invalid-name
                           (int(self._scaling_size * stretch_factor[0]),
                            int(self._scaling_size * stretch_factor[1])),
                           'bilinear') - self._mean).transpose((2, 0, 1))
            retimx = (im.shape[2] - net.blobs[self._layer_name].data.shape[3]) // 2
            retimy = (im.shape[1] - net.blobs[self._layer_name].data.shape[2]) // 2
            retim = im[:,
                       retimy:retimy+net.blobs[self._layer_name].data.shape[2],
                       retimx:retimx+net.blobs[self._layer_name].data.shape[3]]
            sample[...] = retim


def round_to_mbsize(value, batch_size):
    """Round value to multiple of batch size, if required."""
    if value % batch_size == 0:
        return value
    else:
        return value + batch_size - value % batch_size

@click.command()
@click.argument("result_folder", type=click.STRING)  # pylint: disable=no-member
@click.option("--model_name", type=click.STRING,
              help='Model name to use, if a new trial should be created.')
@click.option("--epoch", type=click.INT, default=None,
              help='Epoch to start from, if training is resumed.')
@click.option("--num_epoch", type=click.INT, default=150,
              help='Final number of epochs to reach. Default: 150.')
@click.option("--optimizer_name", type=click.Choice(['adam', 'sgd']),
              default='sgd',
              help='Optimizer to use. Default: sgd.')
@click.option("--lr_param", type=click.FLOAT, default=0.1,
              help='The base learning rate to use. Default: 0.1.')
@click.option("--lr_decay_sched", type=click.STRING, default='90,135',
              help='Scheduled learning rate changes.')
@click.option("--lr_decay_ratio", type=float, default=0.1,
              help='Ratio for the change.')
@click.option("--mom_param", type=click.FLOAT, default=0.9,
              help='The momentum to use if SGD is the optimizer. Default: 0.9.')
@click.option("--wd_param", type=click.FLOAT, default=0.0001,
              help='The weight decay to use. Default: 0.0001.')
@click.option("--monitor", type=click.BOOL, default=False, is_flag=True,
              help='Use extended monitoring (slows down training).')
@click.option("--allow_overwrite", type=click.BOOL, default=False, is_flag=True,
              help='Allow reuse of an existing result directory.')
@click.option("--use_cpu", type=click.BOOL, default=False, is_flag=True,
              help='Use the CPU. If not set, use the GPU.')
# pylint: disable=too-many-arguments, unused-argument
def cli(result_folder,
        model_name=None,
        epoch=None,
        num_epoch=150,
        optimizer_name='sgd',
        lr_param=0.1,
        lr_decay_sched='90,135',
        lr_decay_ratio=0.1,
        mom_param=0.9,
        wd_param=0.0001,
        monitor=False,
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
    from data import _MEAN
    # Setup the output folder, including logging.
    model, optimizer, out_folder, base_iter = _model(
        result_folder,
        tr_data.shape[0],
        model_name,
        epoch,
        10,
        optimizer_name,
        lr_param,
        lr_decay_sched,
        lr_decay_ratio,
        mom_param,
        wd_param,
        False,
        allow_overwrite)
    batch_size = model.blobs['data'].shape[0]
    logger = mnt.JSONLogger(str(out_folder),
                            'model',
                            {'train': ['train_loss', 'train_accuracy'],
                             'test': ['test_loss', 'test_accuracy']},
                            base_iter=base_iter,
                            write_every=round_to_mbsize(10000, batch_size),
                            create_plot=True)
    progr_ind = mnt.ProgressIndicator()
    cropper = RandCropMonitor('data', _MEAN)
    if monitor:
        extra_monitors = [
            mnt.ActivationMonitor(round_to_mbsize(10000, batch_size),
                                  os.path.join(str(out_folder),
                                               'visualizations' + os.sep),
                                  selected_blobs=['data']),
            mnt.ActivationMonitor(round_to_mbsize(10000, batch_size),
                                  os.path.join(str(out_folder),
                                               'visualizations' + os.sep),
                                  selected_blobs=['resblock3_out', 'avpool'],
                                  sample={'data': tr_data[0]}),
            mnt.FilterMonitor(round_to_mbsize(10000, batch_size),
                              os.path.join(str(out_folder),
                                           'visualizations' + os.sep),
                              selected_parameters={'resblock1_conv1': [0],
                                                   'resblock3_conv1': [0],
                                                   'resblock7_conv1': [0]}),
            mnt.GradientMonitor(round_to_mbsize(10000, batch_size),
                                os.path.join(str(out_folder),
                                             'visualizations' + os.sep),
                                relative=True,
                                selected_parameters={'resblock1_conv1': [0, 1],
                                                     'resblock3_conv1': [0, 1],
                                                     'resblock7_conv1': [0, 1]}),
        ]
    else:
        extra_monitors = []
    model.fit(round_to_mbsize(num_epoch * 50000, batch_size),
              optimizer,
              X={'data': tr_data, 'labels': tr_labels},
              X_val={'data': te_data, 'labels': te_labels},
              test_interval=round_to_mbsize(50000, batch_size),
              train_callbacks=[
                  progr_ind,
                  logger,
                  mnt.RotatingMirroringMonitor({'data': 0}, 0, 0.5),
                  cropper,
                  mnt.Checkpointer(os.path.join(str(out_folder),
                                                'model'),
                                   round_to_mbsize(50000 * 10, batch_size),
                                   base_iterations=base_iter),
              ] + extra_monitors,
              test_callbacks=[
                  progr_ind,
                  cropper,
                  logger],
#              test_initialization=True,
             )


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
