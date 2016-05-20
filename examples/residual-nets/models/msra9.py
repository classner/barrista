#!/usr/bin/env python
"""A reimplementation of the 50-layer MSRA residual net."""
# pylint: disable=wrong-import-position, invalid-name, no-member

import logging as _logging
import cv2 as _cv2
import numpy as _np

import barrista.design as _ds

_LOGGER = _logging.getLogger()

_netspec = _ds.NetSpecification([[128, 3, 32, 32], [128,]],
                                inputs=['data', 'labels'],
                                predict_inputs=['data'],
                                predict_input_shapes=[[128, 3, 32, 32]])

_layers = []
_l_idx = 0
_USE_GLOBAL_STATS = None

def ResBlock(n_in, in_name, n_out, stride=1):
    """Create a residual block."""
    global _l_idx  # pylint: disable=global-statement
    layers = []
    layers.append(_ds.ConvolutionLayer(
        name='resblock{}_conv1'.format(_l_idx),
        bottoms=[in_name],
        Convolution_num_output=n_out,
        Convolution_kernel_size=(3, 3),
        Convolution_stride=(stride, stride),
        Convolution_pad=(1, 1),
        Convolution_weight_filler=_ds.PROTODETAIL.FillerParameter(type='msra')
    ))
    layers.append(_ds.BatchNormLayer(
        name='resblock{}_bn1'.format(_l_idx),
        BatchNorm_use_global_stats=_USE_GLOBAL_STATS,
        BatchNorm_moving_average_fraction=0.9
    ))
    layers.append(_ds.ScaleLayer(
        name='resblock{}_scale1'.format(_l_idx),
        Scale_bias_term=True
    ))
    layers.append(_ds.ReLULayer(name='resblock{}_relu1'.format(_l_idx)))
    layers.append(_ds.ConvolutionLayer(
        Convolution_num_output=n_out,
        Convolution_kernel_size=(3, 3),
        Convolution_stride=(1, 1),
        Convolution_pad=(1, 1),
        Convolution_weight_filler=_ds.PROTODETAIL.FillerParameter(type='msra')
    ))
    layers.append(_ds.BatchNormLayer(
        name='resblock{}_bn2'.format(_l_idx),
        BatchNorm_use_global_stats=_USE_GLOBAL_STATS,
        BatchNorm_moving_average_fraction=0.9
    ))
    layers.append(_ds.ScaleLayer(
        name='resblock{}_scale2'.format(_l_idx),
        Scale_bias_term=True
    ))
    sum_in = [in_name, 'resblock{}_scale2'.format(_l_idx)]
    # 'resblock{}_scale2'.format(_l_idx)]
    if n_in != n_out:
        layers.append(_ds.ConvolutionLayer(
            name='resblock{}_sidepath'.format(_l_idx),
            bottoms=[in_name],
            Convolution_num_output=n_out,
            Convolution_kernel_size=(1, 1),
            Convolution_stride=(stride, stride),
            Convolution_pad=(0, 0)
        ))
        sum_in[0] = 'resblock{}_sidepath'.format(_l_idx)
    layers.append(_ds.EltwiseLayer(
        name='resblock{}_sum'.format(_l_idx),
        bottoms=sum_in,
        Eltwise_operation=_ds.PROTODETAIL.EltwiseParameter.SUM))
    layers.append(_ds.ReLULayer(name='resblock{}_out'.format(_l_idx)))
    _l_idx += 1
    return layers, 'resblock{}_out'.format(_l_idx - 1)


def _construct_resnet(blocks_per_part):
    _layers.append(_ds.ConvolutionLayer(
        name='conv_initial',
        bottoms=['data'],
        Convolution_num_output=16,
        Convolution_kernel_size=(3, 3),
        Convolution_stride=(1, 1),
        Convolution_pad=(1, 1),
        Convolution_weight_filler=_ds.PROTODETAIL.FillerParameter(
            type='msra')))
    _layers.append(_ds.BatchNormLayer(
        name='bn_initial',
        BatchNorm_use_global_stats=_USE_GLOBAL_STATS,
        BatchNorm_moving_average_fraction=0.9
    ))
    _layers.append(_ds.ScaleLayer(
        name='scale_initial',
        Scale_bias_term=True
    ))
    _layers.append(_ds.ReLULayer(name='relu_initial'))
    last_out = 'relu_initial'
    for i in range(blocks_per_part):
        layers, last_out = ResBlock(16, last_out, 16)
        _layers.extend(layers)
    for i in range(blocks_per_part):
        layers, last_out = ResBlock(32 if i > 0 else 16,
                                    last_out,
                                    32,
                                    1 if i > 0 else 2)
        _layers.extend(layers)
    for i in range(blocks_per_part):
        layers, last_out = ResBlock(64 if i > 0 else 32,
                                    last_out,
                                    64,
                                    1 if i > 0 else 2)
        _layers.extend(layers)
    _layers.append(_ds.PoolingLayer(
        name='avpool',
        Pooling_kernel_size=8,
        Pooling_stride=1,
        Pooling_pad=0,
        Pooling_pool=_ds.PROTODETAIL.PoolingParameter.AVE))
    _layers.append(_ds.InnerProductLayer(
        InnerProduct_num_output=10,
        InnerProduct_weight_filler=_ds.PROTODETAIL.FillerParameter(
            type='uniform',
            min=-_np.sqrt(2./64.),
            max=_np.sqrt(2./64.)),
        name='net_out'))
    _layers.append(_ds.BatchNormLayer(
        name='net_out_bn',
        BatchNorm_use_global_stats=_USE_GLOBAL_STATS,
        BatchNorm_moving_average_fraction=0.9
    ))
    _layers.append(_ds.ScaleLayer(
        name='net_out_bnscale'))
    _layers.append(_ds.SoftmaxLayer(
        name='score',
        bottoms=['net_out_bnscale'],
        include_stages=['predict']))
    _layers.append(_ds.SoftmaxWithLossLayer(
        name='loss',
        bottoms=['net_out_bnscale', 'labels'],
        include_stages=['fit']))
    _layers.append(_ds.AccuracyLayer(
        name='accuracy',
        bottoms=['net_out_bnscale', 'labels'],
        include_stages=['fit']))

_construct_resnet(9)
_netspec.layers = _layers
MODEL = _netspec.instantiate()
for pname, pval in MODEL.params.items():
    if 'sidepath' in pname:
        w_ary = _np.zeros((pval[0].data.shape[1], pval[0].data.shape[0]),
                          dtype='float32')
        w_ary[:, :pval[0].data.shape[1]] = _np.eye(pval[0].data.shape[1])
        w_ary = w_ary.T
        pval[0].data[:] = w_ary.reshape(pval[0].data.shape)


if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    _LOGGER = _logging.getLogger(__name__)
    name = __file__ + '_vis.png'
    _LOGGER.info("Rendering model to %s.",
                 name)
    vis = MODEL.visualize()
    _cv2.imwrite(name, vis)
    _LOGGER.info("Done.")
