# -*- coding: utf-8 -*-
"""An illustrative example of the usage of `barrista`."""
# pylint: disable=F0401, C0103, E1101
import os
import sys
import cv2
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import numpy as np  # noqa

# This provides us with tools to design a network.
import barrista.design as design  # noqa
from barrista.design import (ConvolutionLayer, ReLULayer, PoolingLayer,
                             DropoutLayer, InnerProductLayer,
                             SoftmaxLayer, SoftmaxWithLossLayer,
                             AccuracyLayer)  # noqa
from barrista.tools import TemporaryDirectory  # noqa
# The monitoring module comes with helpful tools to monitor progress and
# performance.
from barrista.monitoring import (ProgressIndicator, Checkpointer,
                                 JSONLogger)

from barrista import solver as _solver

# When `predict_inputs` and `predict_Input_shapes` are not specified, this
# is used as a straightforward network specification. It they are supplied,
# a virtual second network with stage `predict` is used at prediction time.
netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                  inputs=['data', 'annotations'],
                                  predict_inputs=['data'],
                                  predict_input_shapes=[[10, 3, 51, 51]])

# This is a VGG like convolutional network. This could now even be created
# procedural!
layers = []
conv_params = {'Convolution_kernel_size': 3,
               'Convolution_num_output': 32,
               'Convolution_pad': 1}

layers.append(ConvolutionLayer(**conv_params))
layers.append(ReLULayer())
layers.append(ConvolutionLayer(**conv_params))
layers.append(ReLULayer())
layers.append(PoolingLayer(Pooling_kernel_size=2))
layers.append(DropoutLayer(Dropout_dropout_ratio=0.25))

conv_params['Convolution_num_output'] = 64
layers.append(ConvolutionLayer(**conv_params))
layers.append(ReLULayer())
layers.append(ConvolutionLayer(**conv_params))
layers.append(ReLULayer())
layers.append(PoolingLayer(Pooling_kernel_size=2))
layers.append(DropoutLayer(Dropout_dropout_ratio=0.25))

layers.append(InnerProductLayer(InnerProduct_num_output=256))
layers.append(ReLULayer())
layers.append(DropoutLayer(Dropout_dropout_ratio=0.25))

layers.append(InnerProductLayer(tops=['net_out'], InnerProduct_num_output=10))

# Output layer for stage `predict`.
layers.append(SoftmaxLayer(tops=['out'], include_stages=['predict']))

# Output layers for stage `fit`.
layers.append(SoftmaxWithLossLayer(name='loss',
                                   bottoms=['net_out', 'annotations'],
                                   include_stages=['fit']))
layers.append(AccuracyLayer(name='accuracy',
                            bottoms=['net_out', 'annotations'],
                            include_stages=['fit']))

netspec.layers.extend(layers)
# Create the network. Notice how all layers are automatically wired! If you
# selectively name layers or blobs, this is taken into account.
net = netspec.instantiate()

# Let's do some training (the data does absolutely make no sense and this is
# done solely for illustrative purposes). Note that the amount of inputs may
# be arbitrary, and batching, etc. is automatically taken care of!
X = {'data': np.zeros((11, 3, 51, 51), dtype='float32'),
     'annotations': np.ones((11, 1), dtype='float32')}

with TemporaryDirectory() as tmpdir:
    # Configure our monitors
    # .
    progress = ProgressIndicator()
    perforce = JSONLogger(tmpdir,
                          'test',
                          {'test': ['test_loss',
                                    'test_accuracy'],
                           'train': ['train_loss',
                                     'train_accuracy']})
    checkptr = Checkpointer(os.path.join(tmpdir, 'test_net_'), 50)
    # Run the training.
    net.fit(100,
            _solver.SGDSolver(base_lr=0.01),
            X,
            test_interval=50,  # optional
            X_val=X,  # optional
            train_callbacks=[progress,
                             perforce,
                             checkptr],
            test_callbacks=[progress,
                            perforce])
    # Note the flexibility you have with the monitors: they may be used for any
    # task! By using a different JSON logger for batch- and test-callbacks, you
    # can collect the performance in different logs.

    # Predict some new data. Note, that this is automatically using the weights
    # of the trained net, but in the `predict` layout.
    results = net.predict(np.zeros((30, 3, 51, 51), dtype='float32'),
                          post_batch_callbacks=[ProgressIndicator()])

    # Reloading a model.
    net.load_blobs_from(os.path.join(tmpdir, 'test_net_50.caffemodel'))

    # Visualizing a model. You can add the parameter `display=True` to directly
    # show it.
    # pylint: disable=W0212
    if design._draw is not None:
        viz = netspec.visualize()
        cv2.imwrite(os.path.join(tmpdir, 'test.png'), viz)

    # Going back to medieval age:
    netspec.to_prototxt(output_filename=os.path.join(tmpdir,
                                                     'test.prototxt'))
    netspec_rel = design.NetSpecification.from_prototxt(
        filename=os.path.join(tmpdir, 'test.prototxt'))
