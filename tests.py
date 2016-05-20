"""Unittests for the barrista project."""
# pylint: disable=F0401, C0330, C0302, C0103, R0201, R0914, R0915, W0212
# pylint: disable=no-name-in-module, no-member

import unittest
import logging
logging.basicConfig(level=logging.WARN)

try:
    import cv2  # pylint: disable=W0611
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


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
        _ = design.NetSpecification([[10, 3, 51, 51], [10]],  # noqa
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
        if hasattr(netspec_msg, 'input_shape'):
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
        import copy
        sspec = design.LayerSpecification()  # noqa
        self.assertTrue(sspec == sspec)
        cspec = copy.deepcopy(sspec)
        self.assertTrue(sspec == cspec)
        cspec.include_min_level = 2
        self.assertTrue(sspec != cspec)


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

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        net.fit(10,
                solver,
                X,
                train_callbacks=[fitpi])
        if hasattr(fitpi.pbar, 'finished'):
            # progressbar2 compatibility.
            self.assertEqual(fitpi.pbar.finished, True)

        # For predict.
        predpi = ProgressIndicator()
        net.predict(np.zeros((20, 3, 3, 3)),
                    test_callbacks=[predpi])
        if hasattr(predpi.pbar, 'finished'):
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
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(InnerProductLayer(InnerProduct_num_output=2,
                                        tops=['out']))
        layers.append(SoftmaxWithLossLayer(
            name='loss',
            bottoms=['out', 'annotations']))
        layers.append(AccuracyLayer(name='accuracy',
                                    bottoms=['out', 'annotations']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        dirpath = tempfile.mkdtemp()
        # For fit.
        fitlog = JSONLogger(dirpath,
                            'tmp',
                            {'test': ['test_loss',
                                      'test_accuracy'],
                             'train': ['train_loss',
                                       'train_accuracy']})

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}
        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(30,
                solver,
                X=X,
                X_val=X,
                test_initialization=True,
                test_interval=10,
                train_callbacks=[fitlog],
                test_callbacks=[fitlog])
        # append.
        fitlog = JSONLogger(dirpath,
                            'tmp',
                            {'test': ['test_loss',
                                      'test_accuracy'],
                             'train': ['train_loss',
                                       'train_accuracy']})
        net.fit(30,
                solver,
                X=X,
                X_val=X,
                test_initialization=True,
                test_interval=10,
                train_callbacks=[fitlog],
                test_callbacks=[fitlog])
        with open(os.path.join(dirpath, 'barrista_tmp.json'), 'r') as inf:
            json_load = json.load(inf)
        self.assertIn('train', list(json_load.keys()))
        self.assertIn('test', list(json_load.keys()))
        self.assertEqual(len(json_load['train']), 12)
        self.assertEqual(len(json_load['test']), 16)
        shutil.rmtree(dirpath)
        # Verify values.
        predres = net.predict(X,
                              out_blob_names=['loss', 'accuracy'],
                              allow_train_phase_for_test=True)
        last_test_loss = [dct['test_loss'] for dct in json_load['test']
                          if 'test_loss' in dct.keys() and
                             dct['NumIters'] == 60][0]
        self.assertEqual(last_test_loss, predres['loss'][0])

    def test_StaticDataMonitor(self):
        """Test the static data monitor."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import StaticDataMonitor

        netspec = design.NetSpecification([[3], [3]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()
        tmon = StaticDataMonitor(X={'a': np.array(range(3)),
                                     'b': np.array(range(5, 8))})
        tmon_test = StaticDataMonitor(X={'a': np.array(range(3)),
                                          'b': np.array(range(5, 8))})
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        assert len(tmon.get_parallel_blob_names()) == 2
        kwargs['callback_signal'] = 'initialize_test'
        tmon_test._initialize_test(kwargs)
        kwargs['callback_signal'] = 'pre_fit'
        tmon._pre_fit(kwargs)
        tmon._pre_train_batch({'net': net})
        assert np.all(net.blobs['a'].data[...] == [0, 1, 2])
        assert np.all(net.blobs['b'].data[...] == [5, 6, 7])
        tmon._pre_train_batch({'net': net})
        assert np.all(net.blobs['a'].data[...] == [0, 1, 2])
        assert np.all(net.blobs['b'].data[...] == [5, 6, 7])
        kwargs['callback_signal'] = 'pre_test'
        tmon_test._pre_test(kwargs)
        tmon_test._pre_test_batch({'testnet': net})
        assert np.all(net.blobs['a'].data[...] == [0, 1, 2])
        assert np.all(net.blobs['b'].data[...] == [5, 6, 7])

    def test_CyclingDataMonitor(self):
        """Test the cycling data monitor."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import CyclingDataMonitor

        netspec = design.NetSpecification([[3], [3]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()
        tmon = CyclingDataMonitor(X={'a': list(range(4)),
                                     'b': np.array(range(5, 9))})
        tmon_test = CyclingDataMonitor(X={'a': list(range(4)),
                                          'b': np.array(range(5, 9))})
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        kwargs['callback_signal'] = 'initialize_test'
        with self.assertRaises(Exception):
            tmon._initialize_test(kwargs)
        tmon_test._initialize(kwargs)
        kwargs['callback_signal'] = 'pre_fit'
        tmon._pre_fit(kwargs)
        tmon._pre_train_batch({'net': net})
        assert np.all(net.blobs['a'].data[...] == [0, 1, 2])
        assert np.all(net.blobs['b'].data[...] == [5, 6, 7])
        tmon._pre_train_batch({'net': net})
        assert np.all(net.blobs['a'].data[...] == [3, 0, 1])
        assert np.all(net.blobs['b'].data[...] == [8, 5, 6])
        kwargs['callback_signal'] = 'pre_test'
        tmon_test._pre_test(kwargs)
        tmon_test._pre_test_batch({'testnet': net})
        assert np.all(net.blobs['a'].data[...] == [0, 1, 2])
        assert np.all(net.blobs['b'].data[...] == [5, 6, 7])

    def test_CyclingDataMonitor_only_preload(self):
        """Test the cycling data monitor preload capability."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import CyclingDataMonitor

        netspec = design.NetSpecification([[3], [3]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()

        tmon = CyclingDataMonitor(
            only_preload=['a', 'b'],
            X={'a': list(range(4)),
               'b': np.array(range(5, 9))})
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        kwargs['callback_signal'] = 'pre_fit'
        tmon._pre_fit(kwargs)
        kwargs = {'net': net, 'testnet': net, 'callback_signal': 'pre_batch'}
        tmon._pre_train_batch(kwargs)
        assert np.all(kwargs['data_orig']['a'] == [0, 1, 2])
        assert np.all(kwargs['data_orig']['b'] == [5, 6, 7])
        tmon._pre_train_batch(kwargs)
        assert np.all(kwargs['data_orig']['a'] == [3, 0, 1])
        assert np.all(kwargs['data_orig']['b'] == [8, 5, 6])
        tmon._pre_test_batch(kwargs)
        assert np.all(kwargs['data_orig']['a'] == [2, 3, 0])
        assert np.all(kwargs['data_orig']['b'] == [7, 8, 5])

    def test_CyclingDataMonitor_augmentation(self):
        """Test the cycling data monitor color data augmentation capability."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import CyclingDataMonitor

        netspec = design.NetSpecification([[1, 3, 5, 5], [3]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()

        ddta1 = np.zeros((3, 5, 5))
        ddta1[0, 0, 0] = 1.
        ddta2 = np.zeros((3, 5, 5))
        ddta2[1, 1, 1] = 1.
        tmon = CyclingDataMonitor(
            only_preload=['a', 'b'],
            X={'a': [ddta1, ddta2],
               'b': np.array(range(5, 7))},
            color_data_augmentation_sigmas={'a': 0.1})
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        # Filling of unspecified sigmas.
        self.assertEqual(tmon._color_data_augmentation_sigmas,
                         {'a': 0.1, 'b': 0.})
        # Equal weights for the first two components.
        self.assertEqual(len(tmon._color_data_augmentation_weights), 1)
        self.assertAlmostEqual(tmon._color_data_augmentation_weights['a'][0],
                               tmon._color_data_augmentation_weights['a'][1],
                               delta=0.01)
        # Third one zero.
        self.assertEqual(tmon._color_data_augmentation_weights['a'][2], 0.)
        # Check components: orthogonal first two, zeros for third.
        self.assertEqual(len(tmon._color_data_augmentation_components), 1)
        self.assertEqual(np.dot(
            tmon._color_data_augmentation_components['a'][:, 0].T,
            tmon._color_data_augmentation_components['a'][:, 1]),
                         0.)
        self.assertTrue(
            np.all(
                tmon._color_data_augmentation_components['a'][2, :2] ==
                [0, 0]))
        self.assertTrue(
            np.all(
                tmon._color_data_augmentation_components['a'][:2, 2] ==
                [0, 0]))
        kwargs['callback_signal'] = 'pre_fit'
        tmon._pre_fit(kwargs)
        kwargs = {'net': net, 'testnet': net, 'callback_signal': 'pre_batch'}
        tmon._pre_train_batch(kwargs)
        # Test layerwise application.
        self.assertTrue(np.all(kwargs['data_orig']['a'][0][2] == 0))
        diff0 = ddta1[0, 0, 0] - kwargs['data_orig']['a'][0][0, 0, 0]
        # pylint: disable=superfluous-parens
        # print(np.max(np.abs(ddta1[0] - kwargs['data_orig']['a'][0][0] -
        #                     diff0)))
        self.assertTrue(np.all(np.isclose(
            ddta1[0] - kwargs['data_orig']['a'][0][0],
            diff0, rtol=1e-04, atol=1e-07)))
        diff1 = ddta1[1, 0, 0] - kwargs['data_orig']['a'][0][1, 0, 0]
        self.assertTrue(np.all(np.isclose(
            ddta1[1] - kwargs['data_orig']['a'][0][1],
            diff1, rtol=1e-04, atol=1e-07)))
        diff2 = ddta1[2, 0, 0] - kwargs['data_orig']['a'][0][2, 0, 0]
        self.assertEqual(diff2, 0.)
        self.assertTrue(np.all(np.isclose(
            ddta1[2] - kwargs['data_orig']['a'][0][2],
            diff2, rtol=1e-04, atol=1e-07)))

    def test_CyclingDataMonitor_resizing(self):
        """Test the cycling data monitor resizing capability."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import CyclingDataMonitor
        import barrista.monitoring as bm
        if bm._cv2 is None:
            # OpenCV is not available, so skip the test.
            return

        netspec = design.NetSpecification([[3, 3, 10, 10], [3, 3, 5, 5]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()

        tmon = CyclingDataMonitor(
            X={'a': [np.ones((3, 5, 5))] * 2,
               'b': np.ones((2, 3, 5, 5))},
            input_processing_flags={'a': 'rc', 'b':'rn'})
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        tmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        kwargs = {'net': net, 'testnet': net}
        tmon._pre_train_batch(kwargs)
        assert np.all(net.blobs['a'].data[...] == 1.)
        assert np.all(net.blobs['b'].data[...] == 1.)

    def test_CyclingDataMonitor_padding(self):
        """Test the cycling data monitor padding capability."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import CyclingDataMonitor

        netspec = design.NetSpecification([[3, 3, 10, 10], [3, 3, 7, 7]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()

        tmon = CyclingDataMonitor(
            X={'a': [np.ones((3, 5, 5))] * 2,
               'b': np.ones((2, 3, 5, 5))},
            input_processing_flags={'a': 'p0', 'b':'p2'})
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        tmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        kwargs = {'net': net, 'testnet': net}
        tmon._pre_train_batch(kwargs)
        assert np.sum(net.blobs['a'].data[...]) == 225.
        assert np.sum(net.blobs['b'].data[...]) == 225. + 432.

    def test_ResizingMonitor(self):
        """Test the resizing monitor."""
        import barrista.design as design
        import barrista.solver as sv
        import numpy as np
        from barrista.monitoring import CyclingDataMonitor, ResizingMonitor

        netspec = design.NetSpecification([[1, 3, 5, 5], [1, 1, 5, 5]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        netspec.layers.append(design.ConvolutionLayer(
            Convolution_kernel_size=3,
            Convolution_num_output=1))
        net = netspec.instantiate()

        dmon = CyclingDataMonitor(
            only_preload=['a', 'b'],
            X={'a': [np.zeros((3, 6, 6)), np.zeros((3, 7, 9))],
               'b': [np.ones((1, 6, 6)), np.ones((1, 7, 9))]})
        tmon = ResizingMonitor(
            blobinfos={'a': 1, 'b': 2},
            net_input_size_adjustment_multiple_of=2
        )
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        dmon._initialize_train(kwargs)
        dmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        tmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        kwargs = {'net': net, 'testnet': net}
        dmon._pre_train_batch(kwargs)
        tmon._pre_train_batch(kwargs)
        self.assertEqual(np.sum(net.blobs['a'].data[...]), 39.)
        self.assertEqual(np.sum(net.blobs['b'].data[...]), 36. + 26.)
        dmon._pre_train_batch(kwargs)
        tmon._pre_train_batch(kwargs)
        self.assertEqual(np.sum(net.blobs['a'].data[...]), 0.)
        self.assertEqual(net.blobs['a'].data.shape, (1, 3, 7, 9))
        self.assertEqual(np.sum(net.blobs['b'].data[...]), 63.)
        self.assertEqual(net.blobs['b'].data.shape, (1, 1, 7, 9))
        dmon._pre_test_batch(kwargs)
        tmon._pre_test_batch(kwargs)
        self.assertEqual(np.sum(net.blobs['a'].data[...]), 39.)
        self.assertEqual(np.sum(net.blobs['b'].data[...]), 36. + 26.)
        # Check that the parallel filling works.
        dmon = CyclingDataMonitor(
            only_preload=['a', 'b'],
            X={'a': [np.zeros((3, 6, 6)), np.zeros((3, 7, 9))],
               'b': [np.ones((1, 6, 6)), np.ones((1, 7, 9))]})
        tmon = ResizingMonitor(
            blobinfos={'a': 1, 'b': 2},
            net_input_size_adjustment_multiple_of=2
        )
        net.fit(3, sv.SGDSolver(base_lr=0.01), train_callbacks=[dmon, tmon])
        self.assertEqual(np.sum(net.blobs['a'].data[...]), 39.)
        self.assertEqual(np.sum(net.blobs['b'].data[...]), 36. + 26.)


    def test_ResizingMonitor_fixed_scale(self):
        """Test the resizing monitor scaling capability."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import CyclingDataMonitor, ResizingMonitor
        import barrista.monitoring as bm
        if bm._cv2 is None:
            # OpenCV is not available, so skip the test.
            return

        netspec = design.NetSpecification([[1, 3, 5, 5], [1, 1, 5, 5]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()

        dmon = CyclingDataMonitor(
            only_preload=['a', 'b'],
            X={'a': [np.zeros((3, 6, 6)), np.zeros((3, 7, 9))],
               'b': [np.ones((1, 6, 6)), np.ones((1, 7, 9))]})
        tmon = ResizingMonitor(
            blobinfos={'a': 1, 'b': 2},
            base_scale=2.,
            net_input_size_adjustment_multiple_of=2,
            interp_methods={'a':'c', 'b':'n'}
        )
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        dmon._initialize_train(kwargs)

        dmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        tmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        kwargs = {'net': net, 'testnet': net}
        dmon._pre_train_batch(kwargs)
        tmon._pre_train_batch(kwargs)
        self.assertEqual(np.sum(net.blobs['a'].data[...]), 75.)
        self.assertEqual(np.sum(net.blobs['b'].data[...]), 36.*4. + 50.)
        dmon._pre_train_batch(kwargs)
        tmon._pre_train_batch(kwargs)
        self.assertEqual(np.sum(net.blobs['a'].data[...]),
                         (15.*19.-14.*18.)*3.)
        self.assertEqual(net.blobs['a'].data.shape, (1, 3, 15, 19))
        self.assertEqual(np.sum(net.blobs['b'].data[...]),
                         (15.*19.-14.*18.)*2.+14.*18.)
        self.assertEqual(net.blobs['b'].data.shape, (1, 1, 15, 19))
        dmon._pre_test_batch(kwargs)
        tmon._pre_test_batch(kwargs)
        self.assertEqual(np.sum(net.blobs['a'].data[...]), 75.)
        self.assertEqual(np.sum(net.blobs['b'].data[...]), 36.*4. + 50.)

    def test_ResizingMonitor_random_scale(self):
        """Test the resizing monitor random scale capability."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import CyclingDataMonitor, ResizingMonitor
        import barrista.monitoring as bm
        if bm._cv2 is None:
            # OpenCV is not available, so skip the test.
            return

        netspec = design.NetSpecification([[1, 3, 5, 5], [1, 1, 5, 5]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()

        dmon = CyclingDataMonitor(
            only_preload=['a', 'b'],
            X={'a': [np.zeros((3, 6, 6))],
               'b': [np.ones((1, 6, 6))]})
        tmon = ResizingMonitor(
            blobinfos={'a': 1, 'b': 2},
            base_scale=2.,
            random_change_up_to=0.5,
            net_input_size_adjustment_multiple_of=1,
            interp_methods={'a':'c', 'b':'n'}
        )
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        dmon._initialize_train(kwargs)

        dmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        tmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        kwargs = {'net': net, 'testnet': net}
        scales = []
        np.random.seed(1)
        for _ in range(1000):
            dmon._pre_train_batch(kwargs)
            tmon._pre_train_batch(kwargs)
            scales.append(net.blobs['a'].data.shape[2])
        from scipy.stats import chisquare, itemfreq
        freq = itemfreq(scales)[:, 1]
        _, pvalue = chisquare(freq)
        self.assertTrue(pvalue > 0.1)

    def test_RotatingMirroringMonitor(self):
        """Test the rotating mirroring monitor."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import (CyclingDataMonitor,
                                         RotatingMirroringMonitor)

        netspec = design.NetSpecification([[1, 3, 5, 5], [1, 1, 5, 5]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()

        adata = np.zeros((3, 5, 5))
        adata[:, 0, 0:3] = 1.
        bdata = np.ones((1, 5, 5))
        bdata[:, 0, 0:3] = 0.
        dmon = CyclingDataMonitor(
            X={'a': [adata],
               'b': [bdata]})
        tmon = RotatingMirroringMonitor(
            blobinfos={'a': 2, 'b': 2},
            max_rotation_degrees=90.
        )
        np.random.seed(2748)
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        dmon._initialize_train(kwargs)

        dmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        tmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        kwargs = {'net': net, 'testnet': net}
        dmon._pre_train_batch(kwargs)
        tmon._pre_train_batch(kwargs)
        self.assertEqual(np.sum(net.blobs['a'].data), 54.)
        self.assertTrue(np.all(net.blobs['a'].data[:, :, 2:4, 0] == 1.))
        self.assertEqual(np.sum(net.blobs['b'].data), 31.)
        self.assertTrue(np.all(net.blobs['b'].data[:, :, 2:4, 0] == 0.))

    def test_RotatingMirroringMonitor_mirroring(self):
        """Test the rotating mirroring monitor mirroring capability."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import (CyclingDataMonitor,
                                         RotatingMirroringMonitor)

        netspec = design.NetSpecification([[1, 3, 5, 5], [1, 1, 5, 5]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()

        adata = np.zeros((3, 5, 5))
        adata[:, 0, 0:3] = 1.
        bdata = np.ones((1, 5, 5))
        bdata[:, 0, 0:3] = 0.
        dmon = CyclingDataMonitor(
            X={'a': [adata],
               'b': [bdata]})
        tmon = RotatingMirroringMonitor(
            blobinfos={'a': 2, 'b': 2},
            max_rotation_degrees=0.,
            mirror_prob=0.5
        )
        np.random.seed(2748)
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        dmon._initialize_train(kwargs)

        dmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        tmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        kwargs = {'net': net, 'testnet': net}
        dmon._pre_train_batch(kwargs)
        tmon._pre_train_batch(kwargs)
        self.assertEqual(np.sum(net.blobs['a'].data), np.sum(adata))
        self.assertTrue(np.all(net.blobs['a'].data[:, :, 0, 2:4] == 1.))
        self.assertEqual(np.sum(net.blobs['b'].data), np.sum(bdata))
        self.assertTrue(np.all(net.blobs['b'].data[:, :, 0, 2:4] == 0.))

    def test_RotatingMirroringMonitor_mirroring_swapvalues(self):
        """Test the rotating mirroring monitor mirroring swap capability."""
        import barrista.design as design
        import numpy as np
        from barrista.monitoring import (CyclingDataMonitor,
                                         RotatingMirroringMonitor)

        netspec = design.NetSpecification([[1, 3, 5, 5], [1, 1, 5, 5]],
                                          inputs=['a', 'b'],
                                          phase=design.Phase.TRAIN)
        net = netspec.instantiate()

        adata = np.zeros((3, 5, 5))
        adata[:, 0, 0:3] = 1.
        bdata = np.ones((1, 5, 5))
        bdata[:, 0, 0:3] = 0.
        dmon = CyclingDataMonitor(
            X={'a': [adata],
               'b': [bdata]})
        tmon = RotatingMirroringMonitor(
            blobinfos={'a': 3, 'b': 3},
            max_rotation_degrees=0.,
            mirror_prob=0.5,
            mirror_value_swaps={'a': {1: [(0, 1), (1, 2)]}},
            mirror_layer_swaps={'a': [(1, 2)]}
        )
        np.random.seed(2748)
        kwargs = {'net': net,
                  'testnet': net,
                  'callback_signal': 'initialize_train'}
        tmon._initialize_train(kwargs)
        dmon._initialize_train(kwargs)

        dmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        tmon._pre_fit({'net': net, 'callback_signal': 'pre_fit'})
        kwargs = {'net': net, 'testnet': net}
        dmon._pre_train_batch(kwargs)
        tmon._pre_train_batch(kwargs)
        self.assertEqual(np.sum(net.blobs['a'].data[:, (0, 1), :, :]),
                         np.sum(adata[(0, 2), :, :]))
        self.assertEqual(np.sum(net.blobs['a'].data[:, 2, :, :]),
                         np.sum(adata[1, :, :]+1))
        self.assertTrue(np.all(net.blobs['a'].data[:, (0, 1), 0, 2:4] == 1.))
        self.assertTrue(np.all(net.blobs['a'].data[:, 2, 0, 2:4] == 2.))
        self.assertEqual(np.sum(net.blobs['b'].data), np.sum(bdata))
        self.assertTrue(np.all(net.blobs['b'].data[:, :, 0, 2:4] == 0.))

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
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
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
        chckptr = Checkpointer(dirpath+os.sep, 10)
        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        solver = _solver.SGDSolver(
            base_lr=0.01,
            snapshot_prefix=dirpath+os.sep)
        net.fit(30,
                solver,
                X=X,
                train_callbacks=[chckptr])
        dircontents = os.listdir(dirpath)
        self.assertIn('_iter_2.caffemodel', dircontents)
        self.assertIn('_iter_3.caffemodel', dircontents)
        if hasattr(solver._solver, 'snapshot'):
            self.assertIn('_iter_2.solverstate', dircontents)
            self.assertIn('_iter_3.solverstate', dircontents)
        shutil.rmtree(dirpath)

    def test_GradientMonitor(self):
        """Test the ``GradientMonitor``."""
        try:
            import matplotlib.pyplot as plt
            _ = plt.figure()
        except RuntimeError:
            return
        except ImportError:
            return
        import barrista.design as design
        import numpy as np
        import os
        from barrista.design import (ConvolutionLayer, ReLULayer,
                                     SoftmaxWithLossLayer, InnerProductLayer)
        from barrista.monitoring import GradientMonitor
        from barrista.tools import TemporaryDirectory
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.PROTODETAIL.TRAIN)
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 0,
                       'Convolution_weight_filler': design.PROTODETAIL.FillerParameter(
                           type='xavier')}
        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        layers.append(InnerProductLayer(
            InnerProduct_num_output=2,
            name='net_out',
            InnerProduct_weight_filler=design.PROTODETAIL.FillerParameter(
                type='xavier')))
        layers.append(SoftmaxWithLossLayer(bottoms=['net_out', 'annotations']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()
        # For fit.
        solver = _solver.SGDSolver(
            base_lr=0.01)
        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}
        X['data'][:, 0, 0, 0] = 1.
        with TemporaryDirectory() as tmpdir:
            net.fit(100,
                    solver,
                    X,
                    train_callbacks=[GradientMonitor(10, tmpdir + os.sep)])
            for idx in range(10):
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir,
                    'gradient_hists_{}.png'.format(idx))))
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir,
                    'gradient_magnitude_{}.png'.format(idx))))
            net.fit(100,
                    solver,
                    X,
                    train_callbacks=[GradientMonitor(10,
                                                     tmpdir + os.sep,
                                                     relative=True)])
            for idx in range(10):
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir,
                    'gradient_hists_rel_{}.png'.format(idx))))
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir,
                    'gradient_magnitude_rel_{}.png'.format(idx))))


    def test_ActivationMonitor(self):
        """Test the ``ActivationMonitor``."""
        try:
            import matplotlib.pyplot as plt
            _ = plt.figure()
        except RuntimeError:
            return
        except ImportError:
            return
        import barrista.design as design
        import numpy as np
        import os
        from barrista.design import (ConvolutionLayer, ReLULayer,
                                     SoftmaxWithLossLayer, InnerProductLayer)
        from barrista.monitoring import ActivationMonitor
        from barrista.tools import TemporaryDirectory
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.PROTODETAIL.TRAIN)
        layers = []
        conv_params = {'name': 'conv',
                       'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 0,
                       'Convolution_weight_filler': design.PROTODETAIL.FillerParameter(
                           type='xavier')}
        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        layers.append(InnerProductLayer(
            InnerProduct_num_output=2,
            name='net_out',
            InnerProduct_weight_filler=design.PROTODETAIL.FillerParameter(
                type='xavier')))
        layers.append(SoftmaxWithLossLayer(bottoms=['net_out', 'annotations']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()
        # For fit.
        solver = _solver.SGDSolver(
            base_lr=0.01)
        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}
        X['data'][:, 0, 0, 0] = 1.
        with TemporaryDirectory() as tmpdir:
            net.fit(100,
                    solver,
                    X,
                    train_callbacks=[ActivationMonitor(10, tmpdir + os.sep)])
            for idx in range(10):
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir,
                    'activations_conv_{}.png'.format(idx))))


    def test_FilterMonitor(self):
        """Test the ``FilterMonitor``."""
        try:
            import matplotlib.pyplot as plt
            _ = plt.figure()
        except RuntimeError:
            return
        except ImportError:
            return
        import barrista.design as design
        import numpy as np
        import os
        from barrista.design import (ConvolutionLayer, ReLULayer,
                                     SoftmaxWithLossLayer, InnerProductLayer)
        from barrista.monitoring import FilterMonitor
        from barrista.tools import TemporaryDirectory
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.PROTODETAIL.TRAIN)
        layers = []
        conv_params = {'name': 'conv',
                       'Convolution_kernel_size': 3,
                       'Convolution_num_output': 32,
                       'Convolution_pad': 0,
                       'Convolution_weight_filler': design.PROTODETAIL.FillerParameter(
                           type='xavier')}
        layers.append(ConvolutionLayer(**conv_params))
        layers.append(ReLULayer())
        layers.append(InnerProductLayer(
            InnerProduct_num_output=2,
            name='net_out',
            InnerProduct_weight_filler=design.PROTODETAIL.FillerParameter(
                type='xavier')))
        layers.append(SoftmaxWithLossLayer(bottoms=['net_out', 'annotations']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()
        # For fit.
        solver = _solver.SGDSolver(
            base_lr=0.01)
        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}
        X['data'][:, 0, 0, 0] = 1.
        with TemporaryDirectory() as tmpdir:
            net.fit(100,
                    solver,
                    X,
                    train_callbacks=[FilterMonitor(10, tmpdir + os.sep)])
            for idx in range(10):
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir,
                    'parameters_conv_0_{}.png'.format(idx))))


class NetTestCase(unittest.TestCase):

    """Test the new ``Net`` functions."""

    def test_instantiation(self):
        """Test ``Net`` constructors."""
        import barrista.design as design
        from barrista.design import ConvolutionLayer, ReLULayer, Phase
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
            net = Net(tmpfile.name, Phase.TEST)
        # In older versions of caffe, the input layer was not visible.
        self.assertTrue(len(net.layers) in [2, 3])
        self.assertEqual(net.blobs[net.inputs[0]].data.shape, (10, 3, 3, 3))
        self.assertTrue(net.blobs[net.inputs[1]].data.shape == (10,) or
                        net.blobs[net.inputs[1]].data.shape == (10, 1, 1, 1))

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

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
                X)
        predictions = np.array(net.predict(np.zeros((10, 3, 3, 3))))
        predictions = np.argmax(predictions, axis=1)
        self.assertEqual(np.sum(predictions == 1), 10)
        # Force to use the fit network.
        accy = net.predict(X,
                           use_fit_network=True,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

    def test_reshape_blob(self):
        """Test the reshaping of a blob across nets."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer,
                                     SoftmaxLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10, 1, 3, 3]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1,
                       'name': 'out'}

        layers.append(ConvolutionLayer(**conv_params))
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
        net.reshape_blob('data', 10, 3, 5, 5)
        net.blobs['annotations'].reshape(10, 1, 5, 5)

        X = {'data': np.zeros((10, 3, 5, 5), dtype='float32'),
             'annotations': np.ones((10, 1, 5, 5), dtype='float32')}

        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
                X)
        predictions = np.array(net.predict(np.zeros((10, 3, 5, 5))))
        predictions = np.argmax(predictions, axis=1)
        self.assertEqual(np.sum(predictions == 1), 250)
        # Force to use the fit network.
        accy = net.predict(X,
                           use_fit_network=True,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

    def test_load_blobs_from(self):
        """Test the loading method."""
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
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(InnerProductLayer(name='outlbf',
                                        InnerProduct_num_output=2,
                                        tops=['out']))
        layers.append(SoftmaxWithLossLayer(bottoms=['out', 'annotations']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        dirpath = tempfile.mkdtemp()
        chckptr = Checkpointer(dirpath + os.sep, 10)
        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        solver = _solver.SGDSolver(
            base_lr=0.01,
            snapshot_prefix=dirpath+os.sep)
        net.fit(20,
                solver,
                X=X,
                train_callbacks=[chckptr])
        checkp0_data = net.params['_layer_0'][0].data.copy()
        net.params['_layer_0'][0].data[...] = 10.
        assert np.any(net.params['_layer_0'][0].data != checkp0_data)
        net.load_blobs_from(os.path.join(dirpath, '_iter_2.caffemodel'))
        assert np.all(net.params['_layer_0'][0].data == checkp0_data)
        if (hasattr(solver._solver, 'restore') and
            hasattr(solver._solver, 'snapshot')):
            # Check for newer versions of caffe the solver restore method.
            solver._solver.restore(os.path.join(dirpath, '_iter_2.solverstate'))
        shutil.rmtree(dirpath)

    def test_multiinput(self):
        """Test multiinput prediction."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
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

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
				X)
        accy = net.predict(X,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)
        if CV2_AVAILABLE:
            accy = net.predict(X,
                               input_processing_flags={'data': 'rc',
                                                       'annotations': 'n'},
                               allow_train_phase_for_test=True)['accuracy'][0]
            self.assertEqual(accy, 1.0)
        accy = net.predict(X, input_processing_flags={'data': 'p0',
                                                      'annotations': 'n'},
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

    def test_multioutput(self):
        """Test multioutput prediction."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, EuclideanLossLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 7, 7], [10, 1, 7, 7]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 7, 7]])
        layers = []

        layers.append(ConvolutionLayer(Convolution_kernel_size=3,
                                       Convolution_num_output=1,
                                       Convolution_pad=1,
                                       name='conv1',
                                       tops=['conv1_out']))
        layers.append(ConvolutionLayer(Convolution_kernel_size=3,
                                       Convolution_num_output=1,
                                       Convolution_pad=1,
                                       name='conv2',
                                       tops=['conv2_out'],
                                       bottoms=['data']))
        layers.append(EuclideanLossLayer(name='loss1',
                                         bottoms=['conv1_out', 'annotations'],
                                         include_stages=['fit']))
        layers.append(EuclideanLossLayer(name='loss2',
                                         bottoms=['conv2_out', 'annotations'],
                                         include_stages=['fit']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        X = {'data': np.zeros((10, 3, 7, 7), dtype='float32'),
             'annotations': np.ones((10, 1, 7, 7), dtype='float32')}

        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
				X)
        pred = net.predict([np.zeros((3, 3, 3))],
                            input_processing_flags={'data': 'p0'},
                            output_processing_flags={'conv1_out': 'p0',
                                                     'conv2_out': 'n'})
        assert pred['conv1_out'][0].shape == (1, 3, 3)
        assert pred['conv2_out'][0].shape == (1, 7, 7)
        pred = net.predict([np.zeros((3, 3, 3))],
                           input_processing_flags={'data': 'p0'},
                           output_processing_flags={'conv1_out': 'n',
                                                    'conv2_out': 'p0'})
        assert pred['conv1_out'][0].shape == (1, 7, 7)
        assert pred['conv2_out'][0].shape == (1, 3, 3)

    def test_predict_sliding_window(self):
        """Test the ``predict_sliding_window`` method."""
        if not CV2_AVAILABLE:
            return
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

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        solver = _solver.SGDSolver(
            base_lr=0.01)
        net.fit(20,
                solver,
                X)
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

    def test_predict_sliding_window_eq_out(self):
        """Test the ``predict_sliding_window`` method with full size output."""
        if not CV2_AVAILABLE:
            return
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     EuclideanLossLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3]],
                                          inputs=['data'])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1}

        layers.append(ConvolutionLayer(**conv_params))
        netspec.layers.extend(layers)
        net = netspec.instantiate()
        _ = np.array(net.predict_sliding_window(
            np.zeros((10, 3, 5, 5))))
        _ = np.array(net.predict_sliding_window(
            np.zeros((10, 3, 5, 5)),
            overlap_combine_max=False))

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

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        net.fit(20,
                solver,
                X)
        if CV2_AVAILABLE:
            # Rescaling.
            predictions = np.array(net.predict(np.zeros((10, 3, 1, 1)),
                                   input_processing_flags={'data': 'rl'}))
            predictions = np.argmax(predictions, axis=1)
            self.assertEqual(np.sum(predictions == 1), 10)
        # Padding.
        predictions_padded = np.array(
            net.predict(np.zeros((10, 3, 1, 1)),
                        input_processing_flags={'data': 'p0'}))
        predictions = np.argmax(predictions_padded, axis=1)
        self.assertEqual(np.sum(predictions == 1), 10)
        # out_layer_names.
        predictions = np.array(
            net.predict(np.zeros((10, 3, 1, 1)),
                        input_processing_flags={'data': 'p0'},
                        out_blob_names=['out'],
                        input_size_spec=(10, 10)))
        predictions = np.argmax(predictions, axis=1)
        self.assertEqual(np.sum(predictions == 1), 10)
        # Static inputs.
        predictions = np.array(net.predict(
            {'data': np.zeros((10, 3, 3, 3))},
            static_inputs=['data']))
        predictions = np.argmax(predictions, axis=1)
        self.assertEqual(np.sum(predictions == 1), 10)
        # Upscaling.
        _ = np.array(
            net.predict(np.zeros((10, 3, 1, 1)),
                        input_processing_flags={'data': 'p0'},
                        output_processing_flags={'softmax': 'p0'}))
        # Oversample.
        predictions = np.array(net.predict(np.zeros((10, 3, 1, 1)),
                                           oversample=True))
        np.testing.assert_allclose(predictions, predictions_padded, rtol=1e-05)
        if CV2_AVAILABLE:
            predictions = np.array(net.predict(np.zeros((10, 3, 1, 1)),
                                               oversample=True,
                                               before_oversample_resize_to=(5, 5)))
            np.testing.assert_allclose(predictions, predictions_padded, rtol=1e-05)

    def test_predict_upscaling(self):
        """Test the ``predict`` method upscaling capability."""
        if not CV2_AVAILABLE:
            return
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, SoftmaxLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10, 1, 1, 1]],
                                          inputs=['data', 'annotations'],
                                          predict_inputs=['data'],
                                          predict_input_shapes=[[10, 3, 3, 3]])
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 0,
                       'name': 'out'}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(SoftmaxLayer(bottoms=['out'],
                                   include_stages=['predict'],
                                   name='softmax'))
        layers.append(SoftmaxWithLossLayer(bottoms=['out', 'annotations'],
                                           include_stages=['fit']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        solver = _solver.SGDSolver(
            base_lr=0.01)

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        net.fit(20,
                solver,
                X)
        # Upscaling.
        predictions_us = np.array(
            net.predict(np.zeros((10, 3, 1, 1)),
                        input_processing_flags={'data': 'p0'},
                        output_processing_flags={'softmax': 'p0'}))
        self.assertTrue(np.all(predictions_us.shape == (10, 3, 1, 1)))

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
                               'examples/showcase.py'])


class SolverTestCase(unittest.TestCase):

    """Test the tools module."""

    def test_fit(self):
        """Test the fit function."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
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

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        solver = _solver.SGDSolver(base_lr=0.01)
        solver.fit(20,
                   net=net,
                   X=X)
        accy = net.predict(X,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

        new_net = netspec.instantiate()
        new_solver = _solver.SGDSolver(net=new_net,
                                       base_lr=0.01)
        new_solver.fit(20,
                       X)
        accy = new_net.predict(X,
                               allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

        new_net = netspec.instantiate()
        new_solver = _solver.SGDSolver(net=new_net,
                                       base_lr=0.01)
        new_solver.fit(20,
                       X,
                       use_fit_phase_for_validation=True)
        accy = new_net.predict(X,
                               allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

        new_solver.fit(20,
                       X,
                       X_val=X,
                       test_initialization=True,
                       test_interval=10,
                       use_fit_phase_for_validation=True)
        accy = new_net.predict(X,
                               allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

    def test_restore(self):
        """Test the ``restore`` method."""
        import tempfile
        import shutil
        import os
        import barrista.design as design
        import numpy as np
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, PROTODETAIL)
        from barrista.monitoring import Checkpointer
        from barrista import solver as _solver
        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
        layers = []
        conv_params = {'Convolution_kernel_size': 3,
                       'Convolution_num_output': 3,
                       'Convolution_pad': 1,
                       'Convolution_weight_filler':
                       PROTODETAIL.FillerParameter(type='xavier')}

        layers.append(ConvolutionLayer(**conv_params))
        layers.append(InnerProductLayer(name='outlbf',
                                        InnerProduct_num_output=2,
                                        tops=['out']))
        layers.append(SoftmaxWithLossLayer(bottoms=['out', 'annotations']))
        netspec.layers.extend(layers)
        net = netspec.instantiate()

        dirpath = tempfile.mkdtemp()
        chckptr = Checkpointer(dirpath + os.sep, 10)
        X = {'data': np.ones((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        solver = _solver.SGDSolver(
            base_lr=0.01,
            snapshot_prefix=dirpath+os.sep)
        net.fit(30,
                solver,
                X=X,
                train_callbacks=[chckptr])
        if not (hasattr(solver._solver, 'restore') and
                hasattr(solver._solver, 'snapshot')):
            return
        newsolver = _solver.SGDSolver(
            base_lr=0.01,
            snapshot_prefix=dirpath+os.sep)
        newnet = netspec.instantiate()
        newsolver.restore(os.path.join(dirpath, '_iter_2.solverstate'),
                          newnet)
        newsolver.fit(10,
                      X=X)
        self.assertTrue(np.all(net.params['_layer_0'][0].data[...] ==
                               newnet.params['_layer_0'][0].data[...]))
        shutil.rmtree(dirpath)

    def test_sgd(self):
        """Test the stochastic gradient descent."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)

        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
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

        #######################################################################
        # test sgd solver
        #######################################################################
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
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 1)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))
        params = {'net': net, 'base_lr': 2}
        if 'iter_size' in solver_parameter_dict.keys():
            params['iter_size'] = 2
        tmp_instance = tmp(**params)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))
        params['regularization_type'] = 'L1'
        tmp_instance = tmp(**params)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L1')
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))
        if 'iter_size' in solver_parameter_dict.keys():
            params['iter_size'] = 3
        params['regularization_type'] = '--'
        with self.assertRaises(AssertionError):
            _ = tmp(**params).Get_parameter_dict()

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

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        solver = tmp(base_lr=0.01)
        solver.fit(20,
                   X,
                   net=net)
        accy = net.predict(X,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

        solver = tmp(net=net,
                     base_lr=0.01)

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        net.fit(20,
                solver,
                X)
        accy = net.predict(X,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

    def test_nesterov(self):
        """Test the nesterov solver."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)

        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
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

        #######################################################################
        # test nesterov solver
        #######################################################################
        from barrista import solver as _solver
        tmp = _solver.Get_solver_class('nesterov')
        self.assertTrue(issubclass(tmp, _solver.NesterovSolver))
        tmp = _solver.Get_caffe_solver_class(_solver.SolverType.NESTEROV)
        self.assertTrue(issubclass(tmp, _solver.NesterovSolver))

        with self.assertRaises(TypeError):
            tmp(2)

        with self.assertRaises(Exception):
            tmp(iter_size=2)

        tmp_instance = tmp(base_lr=2)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 1)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['momentum'], 0.0)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        params = {'net': net, 'base_lr': 2}
        if 'iter_size' in solver_parameter_dict.keys():
            params['iter_size'] = 2
        tmp_instance = tmp(**params)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['momentum'], 0.0)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        params['regularization_type'] = 'L1'
        tmp_instance = tmp(**params)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L1')
        self.assertEqual(solver_parameter_dict['momentum'], 0.0)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        params['momentum'] = 1.
        tmp_instance = tmp(**params)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L1')
        self.assertEqual(solver_parameter_dict['momentum'], 1.0)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        if 'iter_size' in solver_parameter_dict.keys():
            params['iter_size'] = 3
        params['regularization_type'] = '--'
        del params['momentum']
        with self.assertRaises(AssertionError):
            _ = tmp(**params).Get_parameter_dict()

        del params['regularization_type']
        params['lr_policy'] = 'step'
        with self.assertRaises(AssertionError):
            _ = tmp(**params).Get_parameter_dict()
        params['lr_policy'] = 'xx'
        with self.assertRaises(AssertionError):
            _ = tmp(**params).Get_parameter_dict()
        params['lr_policy'] = 'exp'
        with self.assertRaises(AssertionError):
            _ = tmp(**params).Get_parameter_dict()
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

        solver = tmp(base_lr=0.01,
                     momentum=0.95)

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        net.fit(20,
                solver,
                X)
        accy = net.predict(X,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

    def test_rmsprop(self):
        """Test the RMSProp solver."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)

        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
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

        #######################################################################
        # test rmsprop solver
        #######################################################################
        from barrista import solver as _solver
        if not hasattr(_solver.SolverType, 'RMSPROP'):
            return
        tmp = _solver.Get_solver_class('rmsprop')
        self.assertTrue(issubclass(tmp, _solver.RMSPropSolver))
        tmp = _solver.Get_caffe_solver_class(_solver.SolverType.RMSPROP)
        self.assertTrue(issubclass(tmp, _solver.RMSPropSolver))

        with self.assertRaises(TypeError):
            tmp(2)

        with self.assertRaises(Exception):
            tmp(iter_size=2)

        with self.assertRaises(Exception):
            tmp(base_lr=2)

        with self.assertRaises(Exception):
            tmp(base_lr=2,
                delta=0.1)

        tmp_instance = tmp(base_lr=2,
                           delta=0.1,
                           rms_decay=0.9)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        self.assertEqual(solver_parameter_dict['iter_size'], 1)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['rms_decay'], 0.9)
        self.assertEqual(solver_parameter_dict['delta'], 0.1)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        tmp_instance = tmp(net=net,
                           base_lr=2,
                           delta=0.1,
                           rms_decay=0.9,
                           iter_size=2)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['rms_decay'], 0.9)
        self.assertEqual(solver_parameter_dict['delta'], 0.1)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        tmp_instance = tmp(net=net,
                           base_lr=2,
                           delta=0.1,
                           rms_decay=0.9,
                           iter_size=2,
                           regularization_type='L1')
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L1')
        self.assertEqual(solver_parameter_dict['rms_decay'], 0.9)
        self.assertEqual(solver_parameter_dict['delta'], 0.1)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    delta=0.1,
                    rms_decay=0.9,
                    iter_size=3,
                    regularization_type='--').Get_parameter_dict()

        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    delta=0.1,
                    rms_decay=0.9,
                    lr_policy='step').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    delta=0.1,
                    rms_decay=0.9,
                    lr_policy='xx').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    delta=0.1,
                    rms_decay=0.9,
                    lr_policy='exp').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    delta=0.1,
                    rms_decay=0.9,
                    lr_policy='inv').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    delta=0.1,
                    rms_decay=0.9,
                    lr_policy='multistep').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,
                    base_lr=2,
                    delta=0.1,
                    rms_decay=0.9,
                    lr_policy='poly').Get_parameter_dict()
        with self.assertRaises(AssertionError):
            _ = tmp(net=net,   # noqa
                    base_lr=2,
                    delta=0.1,
                    rms_decay=0.9,
                    lr_policy='sigmoid').Get_parameter_dict()

        solver = tmp(base_lr=2,
                     delta=0.1,
                     rms_decay=0.9)

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        net.fit(20,
                solver,
                X)
        accy = net.predict(X,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

    def test_adadelta(self):
        """Test the Adadelta solver."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)

        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
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

        #######################################################################
        # test AdaDelta solver
        #######################################################################
        from barrista import solver as _solver
        if not hasattr(_solver.SolverType, 'ADADELTA'):
            return
        tmp = _solver.Get_solver_class('adadelta')
        self.assertTrue(issubclass(tmp, _solver.AdaDeltaSolver))
        tmp = _solver.Get_caffe_solver_class(_solver.SolverType.ADADELTA)
        self.assertTrue(issubclass(tmp, _solver.AdaDeltaSolver))

        with self.assertRaises(TypeError):
            tmp(2)

        with self.assertRaises(Exception):
            tmp(iter_size=2)

        with self.assertRaises(Exception):
            tmp(base_lr=2)

        with self.assertRaises(Exception):
            tmp(base_lr=2,
                delta=0.1)

        tmp_instance = tmp(base_lr=2,
                           momentum=0.9)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 1)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['momentum'], 0.9)
        self.assertEqual(solver_parameter_dict['delta'], 1E-8)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        params = {'net': net, 'base_lr': 2, 'momentum': 0.9, 'delta': 0.1}
        if 'iter_size' in solver_parameter_dict.keys():
            params['iter_size'] = 2
        tmp_instance = tmp(**params)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['momentum'], 0.9)
        self.assertEqual(solver_parameter_dict['delta'], 0.1)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        solver = tmp(base_lr=0.001,
                     momentum=0.9)

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        net.fit(20,
                solver,
                X)
        accy = net.predict(X,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

    def test_adagrad(self):
        """Test the AdaGrad solver."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)

        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
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

        #######################################################################
        # test AdaGrad solver
        #######################################################################
        from barrista import solver as _solver
        if not hasattr(_solver.SolverType, 'ADAGRAD'):
            return
        tmp = _solver.Get_solver_class('adagrad')
        self.assertTrue(issubclass(tmp, _solver.AdagradSolver))
        tmp = _solver.Get_caffe_solver_class(_solver.SolverType.ADAGRAD)
        self.assertTrue(issubclass(tmp, _solver.AdagradSolver))

        with self.assertRaises(TypeError):
            tmp(2)

        with self.assertRaises(Exception):
            tmp(iter_size=2)

        with self.assertRaises(Exception):
            tmp(base_lr=2)

        tmp_instance = tmp(base_lr=2,
                           delta=0.1)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 1)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['delta'], 0.1)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        params = {'net': net, 'base_lr': 2, 'delta': 0.1}
        if 'iter_size' in solver_parameter_dict.keys():
            params['iter_size'] = 2
        tmp_instance = tmp(**params)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['delta'], 0.1)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        solver = tmp(base_lr=0.001,
                     delta=0.1)

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        net.fit(20,
                solver,
                X)
        accy = net.predict(X,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)

    def test_adam(self):
        """Test the ADAM solver."""
        import numpy as np
        import barrista.design as design
        from barrista.design import (ConvolutionLayer, InnerProductLayer,
                                     SoftmaxWithLossLayer, AccuracyLayer)

        netspec = design.NetSpecification([[10, 3, 3, 3], [10]],
                                          inputs=['data', 'annotations'],
                                          phase=design.Phase.TRAIN)
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

        #######################################################################
        # test adam solver
        #######################################################################
        from barrista import solver as _solver
        if not hasattr(_solver.SolverType, 'ADAM'):
            return
        tmp = _solver.Get_solver_class('adam')
        self.assertTrue(issubclass(tmp, _solver.AdamSolver))
        tmp = _solver.Get_caffe_solver_class(_solver.SolverType.ADAM)
        self.assertTrue(issubclass(tmp, _solver.AdamSolver))

        with self.assertRaises(TypeError):
            tmp(2)

        with self.assertRaises(Exception):
            tmp(iter_size=2)

        tmp_instance = tmp(base_lr=2)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 1)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['momentum'], 0.9)
        self.assertEqual(solver_parameter_dict['momentum2'], 0.999)
        self.assertEqual(solver_parameter_dict['delta'], 1E-8)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        params = {'net': net, 'base_lr': 2, 'delta': 0.1}
        if 'iter_size' in solver_parameter_dict.keys():
            params['iter_size'] = 2
        tmp_instance = tmp(**params)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L2')
        self.assertEqual(solver_parameter_dict['momentum'], 0.9)
        self.assertEqual(solver_parameter_dict['momentum2'], 0.999)
        self.assertEqual(solver_parameter_dict['delta'], 0.1)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        params['momentum2'] = 1.
        params['regularization_type'] = 'L1'
        tmp_instance = tmp(**params)
        solver_parameter_dict = tmp_instance.Get_parameter_dict()
        self.assertEqual(solver_parameter_dict['base_lr'], 2)
        if 'iter_size' in solver_parameter_dict.keys():
            self.assertEqual(solver_parameter_dict['iter_size'], 2)
        self.assertEqual(solver_parameter_dict['lr_policy'], 'fixed')
        self.assertEqual(solver_parameter_dict['regularization_type'], 'L1')
        self.assertEqual(solver_parameter_dict['momentum'], 0.9)
        self.assertEqual(solver_parameter_dict['momentum2'], 1.0)
        self.assertEqual(solver_parameter_dict['delta'], 0.1)
        self.assertNotIn('weight_decay', list(solver_parameter_dict.keys()))
        self.assertNotIn('power', list(solver_parameter_dict.keys()))

        solver = tmp(base_lr=0.001)

        X = {'data': np.zeros((10, 3, 3, 3), dtype='float32'),
             'annotations': np.ones((10, 1), dtype='float32')}

        net.fit(20,
                solver,
                X)
        accy = net.predict(X,
                           allow_train_phase_for_test=True)['accuracy'][0]
        self.assertEqual(accy, 1.0)


if __name__ == '__main__':
    unittest.main()
