# -*- coding: utf-8 -*-
"""
The configuration module for barrista.

It is possible to programmatically change the configuration. For this, import
`barrista.config` and edit the values as required before importing any other
of the submodules. The configuration change is then taken into account.
"""

import os as _os

#: This folder must contain the ``caffe`` module and is added to the python
#: path after the first inclusion of the `initialization` module.
if 'CAFFE_PYTHON_FOLDER' in _os.environ.keys():
    CAFFE_PYTHON_FOLDER = _os.environ['CAFFE_PYTHON_FOLDER']
else:
    CAFFE_PYTHON_FOLDER = _os.path.abspath(
        _os.path.join(
            _os.path.dirname(__file__), '..', '..'))

#: This folder contains the file ``upgrade_net_proto_text``.
if 'CAFFE_BIN_FOLDER' in _os.environ.keys():
    CAFFE_BIN_FOLDER = _os.environ['CAFFE_BIN_FOLDER']
else:
    CAFFE_BIN_FOLDER = _os.path.join(CAFFE_PYTHON_FOLDER,
                                     '..', 'build', 'tools')

#: This dictionary specifies the layer types and their configuration
#: parameters. The keys are the layer keys, and the values a list of
#: strings, where each string is the name of a parameter prefixed with
#: `_caffe_pb2.` .
LAYER_TYPES = {'AbsVal': [],
               'Accuracy': ['AccuracyParameter'],
               'ArgMax': ['ArgMaxParameter'],
               'Concat': ['ConcatParameter'],
               'ContrastiveLoss': ['ContrastiveLossParameter'],
               'Convolution': ['ConvolutionParameter'],
               'Data': ['DataParameter'],
               'Dropout': ['DropoutParameter'],
               'DummyData': ['DummyDataParameter'],
               'Embed': ['EmbedParameter'],
               'Eltwise': ['EltwiseParameter'],
               'EuclideanLoss': ['LossParameter'],
               'Exp': ['ExpParameter'],
               'Filter': [],
               'Flatten': ['FlattenParameter'],
               'HDF5Data': ['HDF5DataParameter'],
               'HDF5Output': ['HDF5OutputParameter'],
               'HingeLoss': ['HingeLossParameter'],
               'ImageData': ['ImageDataParameter'],
               'InfogainLoss': ['InfogainLossParameter'],
               'InnerProduct': ['InnerProductParameter'],
               'Log': ['LogParameter'],
               'LRN': ['LRNParameter'],
               # Do not add this layer! It is superfluous with this interface
               # and might just be a source of bugs.
               # 'MemoryData': ['MemoryDataParameter'],
               'MultinomialLogisticLoss': ['LossParameter'],
               'MVN': ['MVNParameter'],
               'Pooling': ['PoolingParameter'],
               'Power': ['PowerParameter'],
               'PReLU': ['PReLUParameter'],
               'Python': ['PythonParameter'],
               'Reduction': ['ReductionParameter'],
               'ReLU': ['ReLUParameter'],
               'Resample': [],
               'Reshape': ['ReshapeParameter'],
               'Sigmoid': ['SigmoidParameter'],
               'SigmoidCrossEntropyLoss': ['LossParameter'],
               'Silence': [],
               'Slice': ['SliceParameter'],
               'Softmax': ['SoftmaxParameter'],
               'SoftmaxWithLoss': ['SoftmaxParameter',
                                   'LossParameter'],
               'Split': [],
               'SPP': ['SPPParameter'],
               'TanH': ['TanHParameter'],
               'Threshold': ['ThresholdParameter'],
               'VariableHingeLoss': ['LossParameter'],
               'WindowData': ['WindowDataParameter']
               }
