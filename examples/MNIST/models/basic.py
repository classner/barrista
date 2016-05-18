#!/usr/bin/env python
"""A simple network to run on MNIST."""
# pylint: disable=wrong-import-position, invalid-name, no-member

import logging as _logging
import cv2 as _cv2
import numpy as _np

import barrista.design as _ds

_LOGGER = _logging.getLogger()

_netspec = _ds.NetSpecification([[100, 1, 28, 28], [100,]],
                                inputs=['data', 'labels'],
                                predict_inputs=['data'],
                                predict_input_shapes=[[100, 1, 28, 28]])

_layers = []

# Build the network.
_layers.append(_ds.ConvolutionLayer(
    name='conv1',
    bottoms=['data'],
    Convolution_num_output=32,
    Convolution_kernel_size=(3, 3),
    Convolution_weight_filler=_ds.PROTODETAIL.FillerParameter(
        type='uniform',
        min=-_np.sqrt(1./(3.*3.*1.)),
        max=_np.sqrt(1./(3.*3.*1.)))))
_layers.append(_ds.ReLULayer())
_layers.append(_ds.PoolingLayer(
    Pooling_kernel_size=2,
    Pooling_stride=2))
_layers.append(_ds.ConvolutionLayer(
    name='conv2',
    Convolution_num_output=32,
    Convolution_kernel_size=(3, 3),
    Convolution_weight_filler=_ds.PROTODETAIL.FillerParameter(
        type='uniform',
        min=-_np.sqrt(1./(3.*3.*32.)),
        max=_np.sqrt(1./(3.*3.*32.)))))
_layers.append(_ds.ReLULayer())
_layers.append(_ds.PoolingLayer(
    Pooling_kernel_size=2,
    Pooling_stride=2))
_layers.append(_ds.InnerProductLayer(
    name='out_ip1',
    InnerProduct_num_output=256,
    InnerProduct_weight_filler=_ds.PROTODETAIL.FillerParameter(
        type='uniform',
        min=-_np.sqrt(1./1152.),
        max=_np.sqrt(1./1152.))))
_layers.append(_ds.ReLULayer())
_layers.append(_ds.InnerProductLayer(
    InnerProduct_num_output=10,
    name='net_out',
    tops=['net_out'],
    InnerProduct_weight_filler=_ds.PROTODETAIL.FillerParameter(
        type='uniform',
        min=-_np.sqrt(1./256.),
        max=_np.sqrt(1./256.))))

_layers.append(_ds.SoftmaxLayer(
    name='score',
    bottoms=['net_out'],
    include_stages=['predict']))
_layers.append(_ds.SoftmaxWithLossLayer(
    name='loss',
    bottoms=['net_out', 'labels'],
    include_stages=['fit']))
_layers.append(_ds.AccuracyLayer(
    name='accuracy',
    bottoms=['net_out', 'labels'],
    include_stages=['fit']))


_netspec.layers = _layers

MODEL = _netspec.instantiate()


if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    _LOGGER = _logging.getLogger(__name__)

    name = __file__ + '_vis.png'
    _LOGGER.info("Rendering model to %s.",
                 name)

    vis = MODEL.visualize()
    _cv2.imwrite(name, vis)

    _LOGGER.info("Done.")
