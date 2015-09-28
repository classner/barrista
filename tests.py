"""Unittests for the barrista project."""
from __future__ import print_function
import unittest
import logging
logging.basicConfig(level=logging.INFO)


class NetSpecificationTestCase(unittest.TestCase):

    """Tests the :py:class:`barrista.design.NetSpecification` class."""

    def test_initialization(self):
        """Test initialization and checks."""
        import barrista.design as design
        # Basic init works.
        _ = design.NetSpecification([[2, 2]])
        _ = design.NetSpecification([[2, 2, 2, 2]])
        # Checks work.
        with self.assertRaises(AssertionError):
            _ = design.NetSpecification([[2, 2, 2, 2], [2, 2]])
        with self.assertRaises(AssertionError):
            _ = design.NetSpecification([[2, 2]],
                                        predict_inputs=['test'])
        with self.assertRaises(AssertionError):
            _ = design.NetSpecification([[2, 2]],
                                        predict_input_shapes=[[2, 2]])
        with self.assertRaises(AssertionError):
            _ = design.NetSpecification([[2, 2]],
                                        predict_inputs=['test'],
                                        predict_input_shapes=[[]])
        _ = design.NetSpecification([[10, 3, 51, 51], [10]],              # noqa
                                    inputs=['data', 'annotations'],
                                    predict_inputs=['data'],
                                    predict_input_shapes=[[10, 3]])

    def test_get_predict_net_specification(self):
        """Test the method ``get_predict_net_specification``."""
        import barrista.design as design
        with self.assertRaises(AssertionError):
            netspec = design.NetSpecification([[2, 2]])
            netspec.get_predict_net_specification()
        netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3]])
        pred_netspec = netspec.get_predict_net_specification()
        self.assertEqual(pred_netspec.name, netspec.name)
        self.assertEqual(pred_netspec.debug_info, netspec.debug_info)
        self.assertEqual(pred_netspec.stages, ['predict'])
        self.assertEqual(pred_netspec.level, netspec.level)
        self.assertEqual(pred_netspec.phase, design.Phase.TEST)
        self.assertEqual(pred_netspec.force_backward, False)
        self.assertEqual(pred_netspec.layers, netspec.layers)
        self.assertEqual(pred_netspec.inputs, netspec.predict_inputs)
        self.assertEqual(pred_netspec.input_shape,
                         netspec.predict_input_shapes)

    def test_to_pbuf_message(self):
        """Test the method ``to_pbuf_message``."""
        import barrista.design as design
        from barrista.design import ConvolutionLayer, ReLULayer
        netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        netspec.layers.extend(layers)

        netspec_msg = netspec.to_pbuf_message()
        self.assertEqual(netspec_msg.IsInitialized(), True)
        self.assertEqual(netspec_msg.input, netspec.inputs)
        for msgshape, specshape in zip(netspec_msg.input_shape,
                                       netspec.input_shape):
            self.assertEqual(list(msgshape.dim), specshape)
        self.assertEqual(len(netspec_msg.layer), len(netspec.layers))
        self.assertEqual(netspec_msg.state.phase, netspec.phase)
        self.assertEqual(netspec_msg.state.level, netspec.level)
        self.assertEqual(netspec_msg.state.stage, netspec.stages)
        self.assertEqual(netspec_msg.name, netspec.name)
        self.assertEqual(netspec_msg.debug_info, netspec.debug_info)

    def test_prototxt_conversion(self):
        """Test the prototxt conversion methods."""
        import barrista.design as design
        from barrista.design import ConvolutionLayer, ReLULayer
        import tempfile
        netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        netspec.layers.extend(layers)
        _ = netspec.instantiate()

        netspec_rl = design.NetSpecification.from_prototxt(
            netspec.to_prototxt())

        # Since we have the test for `to_pbuf_message`, we can assume the
        # conversion to prototxt works correctly.
        self.assertEqual(netspec_rl.to_prototxt(), netspec.to_prototxt())

        # Test file io.
        with tempfile.NamedTemporaryFile(mode='r',
                                         suffix=".prototxt") as tmpfile:
            netspec.to_prototxt(output_filename=tmpfile.name)
            tmpfile.file.flush()
            netspec_rl = design.NetSpecification.from_prototxt(
                filename=tmpfile.name)
        # Test instantiation of a loaded net.
        _ = netspec_rl.instantiate()  # noqa

    def test_copy(self):
        """Test the method ``copy``."""
        import barrista.design as design
        from barrista.design import ConvolutionLayer, ReLULayer
        netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[2, 3, 2, 2]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        netspec.layers.extend(layers)
        _ = netspec.instantiate()
        netptext = netspec.to_prototxt()

        netspec_copy = netspec.copy()
        _ = netspec_copy.instantiate()  # noqa
        netcptext = netspec_copy.to_prototxt()
        self.assertEqual(netptext, netcptext)

    def test_visualize(self):
        """Test the ``visualize`` function."""
        import barrista.design as design
        # pylint: disable=W0212
        if design._draw is None:
            return
        from barrista.design import ConvolutionLayer, ReLULayer
        netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[2, 3, 2, 2]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        netspec.layers.extend(layers)
        viz = netspec.visualize()
        self.assertEqual(viz.ndim, 3)

    def test_instantiate(self):
        """Test the method ``instatiate``."""
        import barrista.design as design
        from barrista.design import ConvolutionLayer, ReLULayer
        netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        netspec.layers.extend(layers)
        _ = netspec.instantiate()  # noqa


class LayerSpecificationTestCase(unittest.TestCase):

    """Test the class :py:class:`barrista.design.LayerSpecification`."""

    def test_instantiation(self):
        """Test instantiation."""
        import barrista.design as design
        _ = design.LayerSpecification()  # noqa

    def test_to_pbuf(self):
        """Test protobuf conversion."""
        import barrista.design as design
        layerspec = design.LayerSpecification()
        with self.assertRaises(AssertionError):
            # It is not possible to create an abstract layer without type.
            _ = layerspec.to_pbuf_message(0,  # noqa
                                          None,
                                          ['data'])
        # Check the wiring.
        layerspec.type = 'convolution'
        pbmessage = layerspec.to_pbuf_message(0,
                                              None,
                                              ['data'])
        self.assertEqual(pbmessage.name, '_layer_0')
        self.assertEqual(pbmessage.top[0], '_layer_0')
        self.assertEqual(pbmessage.bottom[0], 'data')
        layerspec2 = design.LayerSpecification()
        layerspec2.type = 'convolution'
        pbmessage2 = layerspec2.to_pbuf_message(1,
                                                layerspec,
                                                ['data'])
        self.assertEqual(pbmessage2.name, '_layer_1')
        self.assertEqual(pbmessage2.top[0], '_layer_1')
        self.assertEqual(pbmessage2.bottom[0], '_layer_0')


class MonitoringTestCase(unittest.TestCase):

    """Test the monitors."""

    def test_ProgressIndicator(self):
        """Test the ``ProgressIndicator``."""
        import barrista.design as design
        import numpy as np
        from barrista.design import ConvolutionLayer, ReLULayer
        from barrista.monitoring import ProgressIndicator
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        netspec.layers.extend(layers)
        net = netspec.instantiate()
        # For fit.
        fitpi = ProgressIndicator()
        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(10,
                solver,
                np.zeros((10, 3, 3, 3)),
                np.ones((10,)),
                after_batch_callbacks=[fitpi])
        self.assertEqual(fitpi.pbar.finished, True)
        # For predict.
        predpi = ProgressIndicator()
        net.predict(np.zeros((10, 3, 3, 3)),
                    after_batch_callbacks=[predpi])
        self.assertEqual(predpi.pbar.finished, True)

    def test_JSONLogger(self):
        """Test the ``JSONLogger``."""
        import tempfile
        import shutil
        import os
        import barrista.design as design
        import numpy as np
        import json
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)
        from barrista.monitoring import JSONLogger
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(InnerProductLayer(InnerProduct_num_output=2,
                                        tops=['out']))
        layers.append(SoftmaxWithLossLayer(bottoms=['out', 'annotations']))
        layers.append(AccuracyLayer(name='accuracy',
                                    bottoms=['out', 'annotations']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        dirpath = tempfile.mkdtemp()
        # For fit.
        fitlog = JSONLogger(dirpath, 'tmp')
        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
                np.zeros((10, 3, 3, 3)),
                np.ones((10,)),
                X_val=np.zeros((10, 3, 3, 3)),
                Y_val=np.ones((10,)),
                test_initialization=True,
                test_interval=10,
                after_batch_callbacks=[fitlog],
                after_test_callbacks=[fitlog])
        with open(os.path.join(dirpath, 'barrista_tmp.json'), 'r') as inf:
            json_load = json.load(inf)
        self.assertIn('train', json_load.keys())
        self.assertIn('test', json_load.keys())
        self.assertEqual(len(json_load['train']), 5)
        self.assertEqual(len(json_load['test']), 6)
        shutil.rmtree(dirpath)

    def test_Checkpointer(self):
        """Test the ``Checkpointer``."""
        import tempfile
        import shutil
        import os
        import barrista.design as design
        import numpy as np
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer)
        from barrista.monitoring import Checkpointer
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(InnerProductLayer(InnerProduct_num_output=2,
                                        tops=['out']))
        layers.append(SoftmaxWithLossLayer(bottoms=['out', 'annotations']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        dirpath = tempfile.mkdtemp()
        chckptr = Checkpointer(dirpath + os.sep, 10)
        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
                np.zeros((10, 3, 3, 3)),
                np.ones((10,)),
                after_batch_callbacks=[chckptr])
        dircontents = os.listdir(dirpath)
        self.assertIn('10.caffemodel', dircontents)
        self.assertIn('20.caffemodel', dircontents)
        shutil.rmtree(dirpath)


class NetTestCase(unittest.TestCase):

    """Test the new ``Net`` functions."""

    def test_instantiation(self):
        """Test ``Net`` constructors."""
        import barrista.design as design
        from barrista.design import ConvolutionLayer, ReLULayer
        from barrista.net import Net
        import tempfile
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        netspec.layers.extend(layers)
        with tempfile.NamedTemporaryFile(mode='r',
                                         suffix=".prototxt") as tmpfile:
            netspec.to_prototxt(output_filename=tmpfile.name)
            tmpfile.file.flush()
            net = Net(tmpfile.name)
        self.assertEqual(len(net.layers), 2)
        self.assertEqual(net.blobs[net.inputs[0]].data.shape, (10, 3, 3, 3))
        self.assertEqual(net.blobs[net.inputs[1]].data.shape, (10,))

    def test_fit(self):
        """Test the fit function."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(InnerProductLayer(InnerProduct_num_output=2,
                                        tops=['out']))
        layers.append(SoftmaxWithLossLayer(bottoms=['out', 'annotations']))
        layers.append(AccuracyLayer(name='accuracy',
                                    bottoms=['out', 'annotations']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()
        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
                np.zeros((10, 3, 3, 3)),
                np.ones((10,)))
        _, accy = net.predict(np.zeros((10, 3, 3, 3)))[0]
        self.assertEqual(accy, 1.0)

    def test_dual_net_use(self):
        """Test the specification of a prediction net."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer,
                                     SoftmaxLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(InnerProductLayer(InnerProduct_num_output=2,
                                        tops=['out']))
        layers.append(SoftmaxLayer(bottoms=['out'],
                                   include_stages=['predict'],
                                   name='softmax'))
        layers.append(SoftmaxWithLossLayer(bottoms=['out', 'annotations'],
                                           include_stages=['fit']))
        layers.append(AccuracyLayer(name='accuracy',
                                    bottoms=['out', 'annotations'],
                                    include_stages=['fit']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
                np.zeros((10, 3, 3, 3)),
                np.ones((10,)))
        predictions = np.array(net.predict(np.zeros((10, 3, 3, 3))))
        predictions = np.argmax(predictions, axis=1)
        self.assertEqual(np.sum(predictions == 1), 10)
        # Force to use the fit network.
        _, accy = np.array(net.predict(np.zeros((10, 3, 3, 3)),
                                       use_fit_network=True))[0]
        self.assertEqual(accy, 1.0)

    def test_predict_sliding_window(self):
        """Test the ``predict_sliding_window`` method."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     EuclideanLossLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(InnerProductLayer(InnerProduct_num_output=1,
                                        tops=['out']))
        layers.append(EuclideanLossLayer(name='se',
                                         bottoms=['out', 'annotations'],
                                         include_stages=['fit']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
                np.zeros((10, 3, 3, 3)),
                np.ones((10,)))
        # Rescaling.
        predictions = np.array(net.predict_sliding_window(
            np.zeros((10, 3, 5, 5))))
        self.assertEqual(np.sum(predictions != 0.), 90)
        # Step size.
        predictions = np.array(net.predict_sliding_window(
            np.zeros((10, 3, 5, 5)),
            extraction_step=(1, 2)))
        self.assertEqual(np.sum(predictions != 0.), 90)
        predictions = np.array(net.predict_sliding_window(
            np.zeros((10, 3, 5, 5)),
            extraction_step=(1, 2),
            account_for_step=False,
            pad_border=False))
        self.assertEqual(np.sum(predictions != 0.), 60)
        predictions = np.array(net.predict_sliding_window(
            np.zeros((10, 3, 5, 5)),
            extraction_step=(1, 2),
            account_for_step=True,
            pad_border=False))
        self.assertEqual(np.sum(predictions != 0.), 90)

    def test_predict(self):
        """Test the ``predict`` method."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, SoftmaxLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(InnerProductLayer(InnerProduct_num_output=2,
                                        tops=['out']))
        layers.append(SoftmaxLayer(bottoms=['out'],
                                   include_stages=['predict'],
                                   name='softmax'))
        layers.append(SoftmaxWithLossLayer(bottoms=['out', 'annotations'],
                                           include_stages=['fit']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
                np.zeros((10, 3, 3, 3)),
                np.ones((10,)))
        # Rescaling.
        predictions = np.array(net.predict(np.zeros((10, 3, 1, 1)),
                               pad_instead_of_rescale=False))
        predictions = np.argmax(predictions, axis=1)
        self.assertEqual(np.sum(predictions == 1), 10)
        # Padding.
        predictions_padded = np.array(net.predict(np.zeros((10, 3, 1, 1)),
                                                  pad_instead_of_rescale=True))
        predictions = np.argmax(predictions_padded, axis=1)
        self.assertEqual(np.sum(predictions == 1), 10)
        # out_layer_names.
        predictions = np.array(net.predict(np.zeros((10, 3, 1, 1)),
                                           pad_instead_of_rescale=True,
                                           out_layer_names=['out']))
        predictions = np.argmax(predictions, axis=1)
        self.assertEqual(np.sum(predictions == 1), 10)
        # Oversample.
        predictions = np.array(net.predict(np.zeros((10, 3, 1, 1)),
                                           pad_instead_of_rescale=True,
                                           oversample=True))
        np.testing.assert_allclose(predictions, predictions_padded, rtol=1e-05)
        predictions = np.array(net.predict(np.zeros((10, 3, 1, 1)),
                                           pad_instead_of_rescale=True,
                                           oversample=True,
                                           before_oversample_resize_to=(5, 5)))
        np.testing.assert_allclose(predictions, predictions_padded, rtol=1e-05)

    def test_visualize(self):
        """Test the ``visualize`` function."""
        import barrista.design as design
        # pylint: disable=W0212
        if design._draw is None:
            return
        from barrista.design import ConvolutionLayer, ReLULayer
        netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[2, 3, 2, 2]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        netspec.layers.extend(layers)
        net = netspec.instantiate()
        viz = net.visualize()
        self.assertEqual(viz.ndim, 3)


class ToolsTestCase(unittest.TestCase):

    """Test the tools module."""

    def test_pad(self):
        """Test the padding function."""
        import numpy as np
        from barrista import tools
        tim = np.ones((3, 1, 1))
        padded = tools.pad(tim, (3, 3))
        aim = np.zeros((3, 3, 3))
        aim[:, 1, 1] = 1.
        self.assertTrue(np.all(aim == padded))
        padded, padding = tools.pad(tim, (3, 3), get_padding=True)
        aim = np.zeros((3, 3, 3))
        aim[:, 1, 1] = 1.
        self.assertTrue(np.all(aim == padded))
        self.assertEqual(padding, ((0, 0), (1., 1.), (1., 1.)))


class ExampleTestCase(unittest.TestCase):

    """Test that the example runs successfully."""

    def test_running(self):
        """Run it."""
        import sys
        import subprocess
        subprocess.check_call([sys.executable,
                               'example.py'])


class SolverTestCase(unittest.TestCase):

    """Test the tools module."""

    def test_sgd(self):
        import barrista.design as design
        from barrista.design import ConvolutionLayer

        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        """Test the sgd solver."""
        from barrista import solver as _solver
        tmp = _solver.Get_solver_class('sgd')
        self.assertTrue(issubclass(tmp, _solver.SGDSolver))
        tmp = _solver.Get_caffe_solver_class(_solver.SolverType.SGD)
        self.assertTrue(issubclass(tmp, _solver.SGDSolver))
        with self.assertRaises(KeyError):
            _ = _solver.Get_solver_class('test')

        with self.assertRaises(TypeError):
            tmp(2)

        with self.assertRaises(Exception):
            tmp(iter_size=2)

        tmp_instance = tmp(base_lr=2)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        self.assertEqual(solver_parameter_dict['iter_size'], 1)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertNotIn('weight_decay', solver_parameter_dict.keys())
        self.assertNotIn('power', solver_parameter_dict.keys())

        tmp_instance = tmp(net=net,
                           base_lr=2,
                           iter_size=2)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertNotIn('weight_decay', solver_parameter_dict.keys())
        self.assertNotIn('power', solver_parameter_dict.keys())

        tmp_instance = tmp(net=net,
                           base_lr=2,
                           iter_size=2,
                           regularization_type='L1')
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L1')
        self.assertNotIn('weight_decay', solver_parameter_dict.keys())
        self.assertNotIn('power', solver_parameter_dict.keys())

        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    iter_size=3,
                    regularization_type='--').Get_parameter_dict()
            
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    lr_policy='step').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    lr_policy='xx').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    lr_policy='exp').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    lr_policy='inv').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    lr_policy='multistep').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    lr_policy='poly').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,   # noqa
                    base_lr=2,
                    lr_policy='sigmoid').Get_parameter_dict()
        
        xx = tmp(net=net,
                 base_lr=2,
                 lr_policy='sigmoid',
                 stepsize=2)
        xx.Get_parameter_dict()


if __name__ == '__main__':
    unittest.main()
