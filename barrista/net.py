# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""Implements an extended Net object."""
from __future__ import print_function

import time as _time
import logging as _logging

import numpy as _np
import cv2 as _cv2
from sklearn.feature_extraction.image import extract_patches as _extract_patches

# CAREFUL! This must be imported before any caffe-related import!
from initialization import init as _init
from tools import chunks as _chunks, pad as _pad, pbufToPyEnum as _pbufToPyEnum

import caffe as _caffe
import caffe.proto.caffe_pb2 as _caffe_pb2

import solver as _solver

_init()
_LOGGER = _logging.getLogger(__name__)


#: Describes the type of the solver used. All solver types supported by caffe
#: are available.
_Phase = _pbufToPyEnum(_caffe_pb2.Phase)


class Net(_caffe.Net):

    r"""
    An extended Net specification.

    ``mode`` specifies the network mode. May be ``caffe.TRAIN`` or
    ``caffe.TEST``. If set to ``caffe.TRAIN``, training is possible, but
    forward propagation is slower and the network is less space efficient than
    when using ``caffe.TEST``.

    :param specification_filename: string.
      The network specification .prototxt file to use.

    :param mode: :py:data:`barrista.design.Phase`.
      The phase to use for this network.

    :param specification: :py:class:`barrista.design.NetSpecification` or None.
      The specification of this network. This cannot be derived solely from
      the .prototxt ``specification_filename``, since this class can express
      more than the caffe protobuf supports. If this is not None, and the
      specification defines ``predict_inputs`` and ``predict_input_shapes``,
      this internally generates a second network with the according inputs
      in stage ``predict`` (with shared weights) and uses it when calling the
      :py:func:`barrista.net.Net.predict` method.
    """

    def __init__(self,
                 specification_filename,
                 mode=_caffe.TEST,
                 specification=None):
        """See class documentation."""
        _caffe.Net.__init__(self, specification_filename, mode)
        self._specification = specification
        self._predict_variant = None
        if self._specification is not None:
            if (self._specification.predict_inputs is not None and
                    self._specification.predict_input_shapes is not None):
                self._predict_variant = self._specification.\
                    get_predict_net_specification().instantiate()
                # Use the blobs of this net.
                self._predict_variant.share_with(self)

    def predict_sliding_window(self,
                               input_sequence,
                               before_batch_callbacks=None,
                               after_batch_callbacks=None,
                               out_layer_names=None,
                               use_fit_network=False,
                               oversample=False,
                               extraction_step=(1, 1),
                               account_for_step=True,
                               interpolation_method=_cv2.INTER_NEAREST,
                               pad_border=True):
        """
        Get predictions for all images in a sliding window manner.

        Similar to the :py:func:`barrista.net.Net.predict` method. See there
        for the parameter descriptions. For this method, every image must be
        at least as big as the input size. It is then sampled using sliding
        window, and for each output layer the reassembled images are returned.

        The output of the network for one patch must be of shape
        (num_layers X 1 X 1) and currently only one output layer is supported.

        :parameter extraction_step: 2-tuple(int).
          Window step size in y and x direction.

        :parameter account_for_step: bool.
          If set to True, the output is resized with nearest neighbor
          interpolation to get a full-sized image.

        :parameter interpolation_method: int in {`cv2.INTER_...`}.
          The interpolation strategy used, if ``account_for_step`` is set and
          the ``extraction_step`` is not ``(1, 1)``.

        :parameter pad_border: bool.
          Whether to return images in the original image size, by adding zero
          padded borders.
        """
        if self._predict_variant is not None and not use_fit_network:
            _LOGGER.info("Using prediction network variant.")
            prednet = self._predict_variant
        else:
            prednet = self
        if extraction_step != (1, 1) and pad_border is True:
            assert account_for_step,\
                ("If ``extraction_step`` != (1, 1) and "
                 "``padborder`` is set, , ``account_for_step`` must be set, "
                 " too.")
        input_dims = prednet.blobs[prednet.inputs[0]].data.shape
        input_image_dims = _np.array(input_dims[2:])
        for im in input_sequence:
            assert(im.shape[0] == input_dims[1])
            assert(im.shape[1] >= input_image_dims[0])
            assert(im.shape[2] >= input_image_dims[1])
        assert (out_layer_names is None or
                len(out_layer_names) == 1), "Only one output layer is supported!"  # noqa
        output_images = []
        for im_id, im in enumerate(input_sequence):
            _LOGGER.info("Processing image %d...", im_id)
            image_beginpoint = _time.time()
            patches = _extract_patches(im,
                                       patch_shape=(input_dims[1],
                                                    input_image_dims[0],
                                                    input_image_dims[1]),
                                       extraction_step=(1,
                                                        extraction_step[0],
                                                        extraction_step[1]))
            sampled_shape = patches.shape[1:3]
            patches = patches.reshape(_np.hstack((_np.prod(patches.shape[:3]),
                                                 patches.shape[3:])))
            results = self.predict(patches,
                                   pad_instead_of_rescale=True,
                                   before_batch_callbacks=before_batch_callbacks,  # noqa
                                   after_batch_callbacks=after_batch_callbacks,
                                   return_unprocessed_outputs=True,
                                   out_layer_names=out_layer_names,
                                   use_fit_network=use_fit_network,
                                   oversample=oversample,
                                   before_oversample_resize_to=None)
            if account_for_step or extraction_step == (1, 1):
                out_im = _np.zeros((results[0].shape[0],
                                    im.shape[1],
                                    im.shape[2]),
                                   dtype=results[0].dtype)
                # Collect the values.
                collected = _np.empty(_np.hstack(([results[0].shape[0]],
                                                  sampled_shape)),
                                      dtype=results[0].dtype)
                for val_idx, val in enumerate(results):
                    try:
                        val.shape = (results[0].shape[0],)
                    except:
                        raise Exception(
                            ("The output shape of the net must be (X, 1, 1) to be "  # noqa
                             "used with the `predict_sliding_window` method. It is "  # noqa
                             "{}.").format(val.shape))
                    collected[:,
                              val_idx // sampled_shape[1],
                              val_idx % sampled_shape[1]] = val[:]
                # Resize.
                for layer_idx in xrange(results[0].shape[0]):
                    layer_area = out_im[
                        layer_idx,
                        int(_np.ceil(input_image_dims[0] / 2.))-1:-int(_np.ceil(input_image_dims[0] / 2.))+1,  # noqa
                        int(_np.ceil(input_image_dims[1] / 2.))-1:-int(_np.ceil(input_image_dims[1] / 2.))+1]  # noqa
                    layer_area[...] = _cv2.resize(
                                        collected[layer_idx],
                                        (layer_area.shape[1],
                                         layer_area.shape[0]),
                                        interpolation=interpolation_method)
                if not pad_border:
                    out_im = out_im[
                        :,
                        int(_np.ceil(input_image_dims[0] / 2.))-1:-int(_np.ceil(input_image_dims[0] / 2.))+1,  # noqa
                        int(_np.ceil(input_image_dims[1] / 2.))-1:-int(_np.ceil(input_image_dims[1] / 2.))+1]  # noqa
                output_images.append(out_im)
            else:
                # Collect the values.
                collected = _np.empty(_np.hstack(([results[0].shape[0]],
                                                  sampled_shape)),
                                      dtype=results[0].dtype)
                for val_idx, val in enumerate(results):
                    try:
                        val.shape = (results[0].shape[0],)
                    except:
                        raise Exception(
                            ("The output shape of the net must be (X, 1, 1) to be "  # noqa
                             "used with the `predict_sliding_window` method. It is "  # noqa
                             "{}.").format(val.shape))
                    collected[:,
                              val_idx // sampled_shape[1],
                              val_idx % sampled_shape[1]] = val[:]
                output_images.append(collected)
            _LOGGER.info("Processed image %d in %ds.",
                         im_id,
                         _time.time() - image_beginpoint)
        return output_images

    def predict(self,
                input_sequence,
                pad_instead_of_rescale=True,
                before_batch_callbacks=None,
                after_batch_callbacks=None,
                return_unprocessed_outputs=False,
                out_layer_names=None,
                use_fit_network=False,
                oversample=False,
                before_oversample_resize_to=None):
        r"""
        Predict samples in the spirit of `scikit-learn`.

        * It is YOUR responsibility to prepare the data in an iterable object
          of numpy arrays with the correctly matching first dimension (i.e.,
          the number of channels).
        * The method will match the data to the input size of the network and
          forward propagate it in batches.

        :param input_sequence: iterable(3D numpy arrays).
          The 3D numpy arrays must match in their first dimension with the
          second dimension of the network input (number of channels). E.g.,
          for a network with input shape [10, 3, 24, 24], you could provide
          inputs as 4D numpy array of shape [100, 3, 10, 10] or a list of
          3D numpy arrays of shape [3, Y, X], where X and Y may be arbitrary
          and different for each image.

        :param pad_instead_of_rescale: bool.
          By default, all inputs are rescaled to the network input size by
          using bicubic interpolation and the full network output is
          returned. If set to ``True``, samples are padded to be in the middle
          of the network input, and depending on whether
          ``return_unprocessed_outputs`` is set or not, you will receive
          the cut-out output areas or the full outputs of the networks
          as result.

        :param before_batch_callbacks: list(callable) or None.
          Each element of this list will be called with the batch id as
          argument before forward propagating a batch.

        :param after_batch_callbacks: list(callable) or None.
          See before, but after a callback.

        :param return_unprocessed_outputs: bool.
          If set to False and ``pad_instead_of_rescale`` is set, returns
          only the relevant areas of the output. Otherwise, returns the full
          network output.

        :param out_layer_names: list(string) or None.
          The names of the layers of which the outputs are returned. If
          unspecified, uses ``self.outputs``. If more than one is used, no
          postprocessing is applied on the outputs!

        :param use_fit_network: bool.
          If set to ``True``, always use this very network, independent of
          whether an internal network in stage ``predict`` is available.
          For more information, see the constructor documentation.

        :param oversample: bool.
          If set to ``True``, uses oversampling and averages the results. You
          have to take care to bring them into the right shape for further
          processing yourself.

        :param before_oversample_resize_to: 2-tuple(int) or None.
          A tuple specifying height and width to resize to before oversampling
          or None if the image should not be resized before oversampling.
        """
        if before_batch_callbacks is None:
            before_batch_callbacks = []
        if after_batch_callbacks is None:
            after_batch_callbacks = []
        if self._predict_variant is not None and not use_fit_network:
            _LOGGER.info("Using prediction network variant.")
            prednet = self._predict_variant
        else:
            prednet = self
        input_dims = prednet.blobs[prednet.inputs[0]].data.shape
        input_image_dims = _np.array(input_dims[2:])
        batch_size = input_dims[0]
        if out_layer_names is None:
            out_layer_names = prednet.outputs
        if oversample:
            assert batch_size % 10 == 0,\
                ("If you use the `oversample` argument as `True`, the batch " +
                 "size must be a multiple of 10!")
        output_images = []
        if before_oversample_resize_to is not None:
            assert (before_oversample_resize_to[0] >= input_image_dims[0] and
                    before_oversample_resize_to[1] >= input_image_dims[1])
        cbparams = dict()
        max_iter = (len(input_sequence) if not oversample
                    else 10 * len(input_sequence))
        cbparams['max_iter'] = max_iter
        cbparams['batch_size'] = batch_size
        cbparams['iter'] = 0
        cbparams['net'] = prednet
        cbparams['X'] = None
        chunk_size = (input_dims[0] if not oversample
                      else input_dims[0] / 10)
        for chunk_id, im_chunk in enumerate(_chunks(input_sequence,
                                                    chunk_size)):
            _LOGGER.info('Preparing chunk %d...', chunk_id)
            net_input = _np.zeros((input_dims[0],
                                   input_sequence[0].shape[0],
                                   input_image_dims[0],
                                   input_image_dims[1]),
                                  dtype=_np.float32)
            for idx, im in enumerate(im_chunk):
                if oversample:
                    if before_oversample_resize_to is not None:
                        oversampling_prep = _cv2.resize(
                            _np.transpose(im, (1, 2, 0)),
                            (before_oversample_resize_to[1],
                             before_oversample_resize_to[0]),
                            interpolation=_cv2.INTER_LINEAR)
                    else:
                        oversampling_prep = _np.transpose(im, (1, 2, 0))
                    net_input[idx * 10:(idx+1) * 10] =\
                        _np.transpose(
                            _caffe.io.oversample([oversampling_prep],
                                                 input_image_dims[:2]),
                            (0, 3, 1, 2))
                else:
                    if pad_instead_of_rescale:
                        net_input[idx] = _pad(im, input_image_dims)
                    else:
                        net_input[idx] = _np.transpose(
                                             _cv2.resize(
                                                 _np.transpose(im, (1, 2, 0)),
                                                 (input_image_dims[1],
                                                  input_image_dims[0]),
                                                 interpolation=_cv2.INTER_CUBIC),  # noqa
                                             (2, 0, 1))
            cbparams['iter'] = (len(output_images) if not oversample
                                else len(output_images) * 10)
            cbparams['X'] = net_input
            for cb in before_batch_callbacks:
                cb(**cbparams)
            to_image = (len(output_images) + input_dims[0] if not oversample
                        else len(output_images) + input_dims[0] / 10)
            _LOGGER.info('Forward propagating chunk %d (image %d to %d of %d)...',  # noqa
                         chunk_id,
                         len(output_images),
                         to_image,
                         len(input_sequence))
            prednet.blobs[prednet.inputs[0]].data[...] = net_input
            forward_prop_beginpoint = _time.time()
            # pylint: disable=W0212
            prednet._forward(0, len(prednet.layers) - 1)
            forward_prop_duration = _time.time() - forward_prop_beginpoint
            _LOGGER.info('Done in %ds.', forward_prop_duration)
            out = {out: prednet.blobs[out].data for out in out_layer_names}

            _LOGGER.info('Extracting output images...')
            if len(out_layer_names) > 1:
                output_images.append([out[outp] for outp in out_layer_names])
                continue
            else:
                full_output_image_blob = out[out_layer_names[0]]
            if full_output_image_blob.ndim == 0:
                output_images.append(full_output_image_blob)
                continue
            target_output_height = input_image_dims[0]
            target_output_width = input_image_dims[1]
            output_dims = _np.array(full_output_image_blob.shape)
            output_image_dims = output_dims[2:]
            if len(output_dims) > 2:
                scale_h, scale_w = (input_image_dims.astype('float') /
                                    output_image_dims.astype('float'))
            else:
                scale_h = scale_w = 1.0
            if not return_unprocessed_outputs and pad_instead_of_rescale:
                if scale_h == 1. and scale_w == 1.:
                    _LOGGER.info('No scaling necessary.')
                    scaling = False
                else:
                    _LOGGER.info('Scale change by %f, %f (h, w).',
                                 scale_h, scale_w)
                    scaling = True
            else:
                scaling = False
            oversampled = []
            for outim in full_output_image_blob:
                if ((len(output_images) == len(input_sequence) and not oversample) or  # noqa
                    ((len(output_images) - len(input_sequence)) < (batch_size / 10) and  # noqa
                     len(oversampled) % 10 == 0 and len(oversampled) / 10 == len(output_images) - len(input_sequence))):  # noqa
                    # The last blob was not completely used.
                    break
                if scaling:
                    outim_resized = _np.empty((output_dims[1],
                                               target_output_height,
                                               target_output_width))
                    for layer_idx, layer in enumerate(outim):
                        outim_resized[layer_idx] = \
                            _cv2.resize(layer,
                                        (int(target_output_width),
                                         int(target_output_height)),
                                        interpolation=_cv2.INTER_CUBIC)
                else:
                    outim_resized = outim
                if not return_unprocessed_outputs and pad_instead_of_rescale \
                   and outim_resized.ndim == 3:  # Otherwise it's not clear...
                    target_image_dims = _np.array(
                        input_sequence[len(output_images)].
                        shape[1:],
                        dtype='int')
                    output_pad_height = (target_output_height -
                                         target_image_dims[0]) / 2.0
                    output_pad_width = (target_output_width -
                                        target_image_dims[1]) / 2.0
                    extracted_work_image = \
                        outim_resized[:,
                                      _np.floor(output_pad_height):
                                      _np.floor(output_pad_height) +
                                      target_image_dims[0],
                                      _np.floor(output_pad_width):
                                      _np.floor(output_pad_width) +
                                      target_image_dims[1]]
                else:
                    extracted_work_image = outim_resized
                if oversample:
                    oversampled.append(extracted_work_image.copy())
                else:
                    output_images.append(extracted_work_image.copy())
            if oversample:
                for os_im_chunk in _chunks(oversampled, 10):
                    chunk_arr = _np.array(os_im_chunk)
                    if chunk_arr.ndim == 4:
                        assert chunk_arr.shape[0] == 10
                        # Flip back the flipped results to average correctly.
                        # The result shape is N, C, H, W, and the second half
                        # of the ten samples are horizontally flipped.
                        chunk_arr[5:] = chunk_arr[5:, :, :, ::-1]
                    output_images.append(_np.mean(chunk_arr, axis=0))
            cbparams['iter'] = (len(output_images) if not oversample
                                else len(output_images) * 10)
            cbparams['out'] = full_output_image_blob
            for cb in after_batch_callbacks:
                cb(**cbparams)
            del cbparams['out']
        for cb in set(before_batch_callbacks + after_batch_callbacks):
            cb.finalize(**cbparams)
        return output_images

    def fit(self,
            iterations,
            solver,
            X=None,
            Y=None,
            X_val=None,
            Y_val=None,
            test_iterations=0,
            test_interval=0,
            test_initialization=False,
            before_batch_callbacks=None,
            after_batch_callbacks=None,
            before_test_callbacks=None,
            after_test_callbacks=None,
            before_test_batch_callbacks=None,
            after_test_batch_callbacks=None,
            read_input_batch_size_from_blob_name=None):
        r"""
        Fit the network to specific data.

        Use monitors from the module :py:mod:`barrista.monitoring` as
        callbacks to monitor the state of the net and create checkpoints.
        This method offers the following kwargs to monitors (* indicates,
        that the values are only available at test time, - indicates, that
        the value is not necessarily available):

        * max_iter,
        * iter,
        * batch_size,
        * loss,
        * accuracy-,
        * net,
        * solver,
        * X,
        * X_val,
        * Y,
        * Y_val,
        * test_loss\*,
        * test_accuracy\*-.

        :param iterations: int.
          The number of training iterations to do. This is the plain number
          of iterations, completely disregarding the batch size, i.e., for
          ``iterations`` being 10 and ``batch_size`` being 10, just one batch
          is forward propagated.

        :param X: iterable(numpy array) or None.
          If specified, is used as input data. It is used sequentially, so
          shuffle it before, if required.

        :param Y: iterable(numpy array) or None.
          Required if ``X`` is specified. The annotations of training.

        :param X_val: iterable(numpy array) or None.
          If specified along with ``Y_val`` and ``test_interval>0``, is used
          to get validation scores.

        :param Y_val: iterable(numpy array) or None.
          The annotation data for the validation set.

        :param test_iterations: int.
          The number of test iterations to determine the validation score,
          if ``test_interval>0``.

        :param test_interval: int.
          The number of iterations between runs on the validation set. Is
          specified in plain iterations, disregarding batch size. Hence, it
          must be a multiple of the batch size.

        :param test_initialization: bool.
          Whether to do a run on the validation set before the training is
          started to get an initial score.

        :param before_batch_callbacks: list(callable).
          List of callback callables. Will be called before a training batch
          is processed. Does not provide ``loss`` and ``accuracy`` kwargs.

        :param after_batch_callbacks: list(callable).
          List of callback callables. Will be called after a training batch
          was processed.

        :param before_test_callbacks: list(callable).
          List of callback callables. Will be called before starting the test
          processing. Does not provide ``loss`` and ``accuracy`` kwargs.

        :param after_test_callbacks: list(callable).
          List of callback callables. Will be called after completing the test
          processing. Does not provide ``loss`` and ``accuracy`` kwargs, but
          provides ``test_loss`` and, if available, ``test_accuracy``.

        :param before_test_batch_callbacks: list(callable).
          List of callback callables. Will be called before a test batch is
          processed. Does not provide ``loss`` and ``accuracy`` kwargs.

        :param after_test_batch_callbacks: list(callable).
          List of callback callables. Will be called after a test batch is
          processed. Does not provide ``loss`` and ``accuracy`` kwargs.

        :param read_input_batch_size_from_blob_name: string.
          The name of the layer to take the input batch size from (as the
          first dimension of its first blob). Must be specified if the
          network does not have explicit inputs (e.g., when trained from
          an LMDB).
        """
        if before_batch_callbacks is None:
            before_batch_callbacks = []
        if after_batch_callbacks is None:
            after_batch_callbacks = []
        if before_test_callbacks is None:
            before_test_callbacks = []
        if after_test_callbacks is None:
            after_test_callbacks = []
        if before_test_batch_callbacks is None:
            before_test_batch_callbacks = []
        if after_test_batch_callbacks is None:
            after_test_batch_callbacks = []
        if len(self.inputs) > 0:
            # Otherwise, a DB backend is used.
            batch_size = self.blobs[self.inputs[0]].data.shape[0]
            assert iterations % batch_size == 0,\
                ('error iterations do not match {} {}'.format(iterations,
                                                              batch_size))
            assert test_interval % batch_size == 0,\
                ('error iterations do not match', test_iterations, batch_size)
            if test_interval != 0 and test_iterations == 0 and X_val is not None:  # noqa
                test_iterations = _np.ceil(float(len(X_val)) /
                                           float(batch_size))
            if read_input_batch_size_from_blob_name is not None:
                assert self.blobs[read_input_batch_size_from_blob_name].data.shape[0] ==\
                    batch_size, ("The input size and the first dimension of "
                                 "the blob to read the batch size from don't "
                                 "match: {}, {}.".format(self.blobs[read_input_batch_size_from_blob_name].data.shape[0],
                                                         batch_size))
        else:
            batch_size = self.blobs[read_input_batch_size_from_blob_name].data.shape[0]  # noqa
        if test_interval > 0:
            assert test_iterations > 0
        # Set the configurable arguments.
        assert test_iterations >= 0
        assert test_interval >= 0

        assert isinstance(solver, _solver.SolverInterface)

        # Training
        # Check the inputs, if supplied.
        if X is not None or Y is not None:
            assert X is not None
            assert Y is not None
            assert len(X) == len(Y)
            assert len(X) >= batch_size
        if test_interval > 0 and X is not None or Y is not None:
            assert X is not None
            assert Y is not None
            assert len(X) == len(Y)
            assert len(X) >= batch_size
        testnet = None
        if test_interval > 0:
            # Setup the test net.
            test_netspec = self._specification.copy()
            test_netspec.phase = _Phase.TEST
            test_netspec.predict_inputs = None
            test_netspec.predict_input_shapes = None
            testnet = test_netspec.instantiate()
            testnet.share_with(self)
        iteration = 0
        curr_sample_pointer = 0
        cbparams = dict()
        cbparams['max_iter'] = iterations
        cbparams['batch_size'] = batch_size
        cbparams['iter'] = 0
        cbparams['loss'] = None
        cbparams['net'] = self
        cbparams['testnet'] = testnet
        cbparams['solver'] = solver
        cbparams['X'] = X
        cbparams['X_val'] = X_val
        cbparams['Y'] = Y
        cbparams['Y_val'] = Y_val
        cbparams['current_callback'] = None
        while iteration <= iterations:
            cbparams['iter'] = iteration
            # Check whether to test the net.
            if (test_interval > 0 and iteration % test_interval == 0 and iteration > 0 or  # noqa
                    iteration == 0 and test_initialization or
                    test_interval > 0 and iteration + batch_size >= iterations):

                cbparams['current_callback'] = 'before_test'
                for cb in before_test_callbacks:
                    cb(**cbparams)

                test_iter = 0
                test_loss = 0.
                test_accuracy = None
                if 'accuracy' in testnet.outputs:
                    test_accuracy = 0.
                test_sample_pointer = 0
                while test_iter < test_iterations:
                    cbparams['current_callback'] = 'before_test_batch'
                    for cb in before_test_batch_callbacks:
                        cb(**cbparams)

                    if X_val is not None:
                        samples = [idx % len(X_val) for idx in
                                   xrange(test_sample_pointer,
                                   test_sample_pointer +
                                   testnet.blobs[testnet.inputs[0]].data.shape[0])]  # noqa
                        test_sample_pointer = ((test_sample_pointer + len(samples)) %  # noqa
                                               len(X_val))
                        testnet.blobs[testnet.inputs[0]].data[...] = \
                            X_val[samples].reshape(
                                testnet.blobs[testnet.inputs[0]].data.shape)
                        testnet.blobs[testnet.inputs[1]].data[...] = \
                            Y_val[samples].reshape(
                                testnet.blobs[testnet.inputs[1]].data.shape)
                    # pylint: disable=W0212
                    # need to multiply by batch_size since it is normalized
                    # internally
                    test_loss += testnet._forward(
                        0, len(testnet.layers) - 1) * batch_size
                    if test_accuracy is not None:
                        # need to multiply by batch_size since it is normalized
                        # internally
                        test_accuracy += float(
                            testnet.blobs['accuracy'].data) * batch_size
                    test_iter += batch_size

                    cbparams['current_callback'] = 'after_test_batch'
                    for cb in after_test_batch_callbacks:
                        cb(**cbparams)

                test_loss /= float(test_iterations)
                cbparams['test_loss'] = test_loss
                if test_accuracy is not None:
                    test_accuracy /= float(test_iterations)
                    cbparams['test_accuracy'] = test_accuracy
                cbparams['current_callback'] = 'after_test'
                for cb in after_test_callbacks:
                    cb(**cbparams)
                del cbparams['test_loss']
                if test_accuracy is not None:
                    del cbparams['test_accuracy']
            if iteration == iterations:
                break

            cbparams['current_callback'] = 'before_batch'
            for cb in before_batch_callbacks:
                cb(**cbparams)

            # If supplied, fill the inputs with the training data.
            if X is not None:
                # Here is the right place to implement random sampling for each
                # SGD iteration.
                samples = [idx % len(X) for idx in xrange(curr_sample_pointer,
                           curr_sample_pointer + batch_size)]
                curr_sample_pointer = ((curr_sample_pointer + len(samples)) %
                                       len(X))
                self.blobs[self.inputs[0]].data[...] = X[samples].reshape(
                    self.blobs[self.inputs[0]].data.shape)
                self.blobs[self.inputs[1]].data[...] = Y[samples].reshape(
                    self.blobs[self.inputs[1]].data.shape)

            loss = cbparams['solver'].Step(1, self)[0]
            cbparams['loss'] = loss
            if 'accuracy' in self.outputs:
                cbparams['accuracy'] = float(self.blobs['accuracy'].data)
            cbparams['iter'] = iteration + batch_size

            cbparams['current_callback'] = 'after_batch'
            for cb in after_batch_callbacks:
                cb(**cbparams)
            del cbparams['loss']
            if 'accuracy' in cbparams.keys():
                del cbparams['accuracy']
            iteration += batch_size
        for cb in set(before_batch_callbacks + after_batch_callbacks +
                      before_test_callbacks + after_test_callbacks):
            cb.finalize(**cbparams)

    def visualize(self,
                  layout_dir='LR',
                  display=False):
        """
        Visualize this model.

        It must have a specification. It has one, if it is an instantiated
        :py:class:`barrista.design.NetSpecification`, of if the parameter
        ``specification`` is set in the constructor.

        :param layout_dir: string in ['LR', 'TB', 'BT'].
          Short string for graph layout direction.

        :param display: bool.
          If set to ``True``, displays the graphic in a window. Press enter
          to close it.

        :returns: 3D numpy array.
          Graphic of the visualization as (H, W, C) image in BGR format.
        """
        if self._specification is not None:
            return self._specification.visualize(layout_dir, display)
        else:
            raise Exception("This model has no specification set!")
