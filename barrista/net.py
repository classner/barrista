# -*- coding: utf-8 -*-
# pylint: disable=E1101, F0401, R0912, R0913, R0914, C0103
"""Implements an extended Net object."""


import time as _time
import logging as _logging

import numpy as _np
from sklearn.feature_extraction.image import extract_patches as _extract_patches

from .tools import chunks as _chunks, pad as _pad

# CAREFUL! This must be imported before any caffe-related import!
from .initialization import init as _init

import caffe as _caffe
try:
    import cv2 as _cv2
    _cv2INTER_CUBIC = _cv2.INTER_CUBIC
    _cv2INTER_LINEAR = _cv2.INTER_LINEAR
    _cv2INTER_NEAREST = _cv2.INTER_NEAREST
    _cv2resize = _cv2.resize
except ImportError:
    _cv2 = None
    _cv2INTER_CUBIC = None
    _cv2INTER_LINEAR = None
    _cv2INTER_NEAREST = None
    _cv2resize = None

_init()
_LOGGER = _logging.getLogger(__name__)
if _cv2 is None:
    _LOGGER.warn('Could not import cv2! Resizing is not available!')


#: Set CPU processing mode.
set_mode_cpu = _caffe.set_mode_cpu
#: Set GPU processing mode.
set_mode_gpu = _caffe.set_mode_gpu


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
                               pre_batch_callbacks=None,
                               post_batch_callbacks=None,
                               out_blob_names=None,
                               use_fit_network=False,
                               oversample=False,
                               extraction_step=(1, 1),
                               account_for_step=True,
                               interpolation_method=_cv2INTER_NEAREST,
                               pad_border=True):
        """
        Get predictions for all images in a sliding window manner.

        Similar to the :py:func:`barrista.net.Net.predict` method. See there
        for the parameter descriptions. For this method, every image must be
        at least as big as the input size. It is then sampled using sliding
        window, and for each output layer the reassembled images are returned.

        The output of the network for one patch must be of shape
        (num_layers X 1 X 1) and currently only one output layer is supported.

        :param input_sequence: iterable(3D numpy arrays)
          The 3D numpy arrays must match in their first dimension with the
          second dimension of the network input (number of channels). E.g.,
          for a network with input shape [10, 3, 24, 24], you could provide
          inputs as 4D numpy array of shape [100, 3, 10, 10] or a list of
          3D numpy arrays of shape [3, Y, X], where X and Y may be arbitrary
          and different for each image. Multi-input networks are not yet
          supported by this method.

        :param pre_batch_callbacks: list(callable) or None.
          Each element of this list will be called with the batch id as
          argument before forward propagating a batch.

        :param post_batch_callbacks: list(callable) or None.
          See before, but after a callback.

        :param out_blob_names: list(string) or None.
          The names of the blobs of which the values are returned. If
          unspecified, uses ``self.outputs``. Must currently contain exactly
          one element.

        :param use_fit_network: bool.
          If set to ``True``, always use this very network, independent of
          whether an internal network in stage ``predict`` is available.
          For more information, see the constructor documentation.

        :param oversample: bool.
          If set to ``True``, uses oversampling and averages the results. You
          have to take care to bring them into the right shape for further
          processing yourself.

        :parameter extraction_step: 2-tuple(int).
          Window step size in y and x direction.

        :parameter account_for_step: bool.
          If set to True, the output is resized with nearest neighbor
          interpolation to get a full-sized image.

        :parameter interpolation_method: int in {cv2.INTER_...}.
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
            assert im.shape[0] == input_dims[1]
            assert im.shape[1] >= input_image_dims[0]
            assert im.shape[2] >= input_image_dims[1]
        assert (out_blob_names is None or
                len(out_blob_names) == 1), "Only one output layer is supported!"  # noqa
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
                                   pre_batch_callbacks=pre_batch_callbacks,  # noqa
                                   post_batch_callbacks=post_batch_callbacks,
                                   out_blob_names=out_blob_names,
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
                for layer_idx in range(results[0].shape[0]):
                    layer_area = out_im[
                        layer_idx,
                        int(_np.ceil(input_image_dims[0] / 2.))-1:
                        -int(_np.ceil(input_image_dims[0] / 2.))+1,
                        int(_np.ceil(input_image_dims[1] / 2.))-1:
                        -int(_np.ceil(input_image_dims[1] / 2.))+1]
                    layer_area[...] = _cv2resize(
                        collected[layer_idx],
                        (layer_area.shape[1],
                         layer_area.shape[0]),
                        interpolation=interpolation_method)
                if not pad_border:
                    out_im = out_im[
                        :,
                        int(_np.ceil(input_image_dims[0] / 2.))-1:
                        -int(_np.ceil(input_image_dims[0] / 2.))+1,
                        int(_np.ceil(input_image_dims[1] / 2.))-1:
                        -int(_np.ceil(input_image_dims[1] / 2.))+1]
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

    def predict(self,  # pylint: disable=R0915
                input_sequence,
                pre_batch_callbacks=None,
                post_batch_callbacks=None,
                out_blob_names=None,
                use_fit_network=None,
                oversample=False,
                before_oversample_resize_to=None,
                input_processing_flags=None,
                output_processing_flags=None,
                static_inputs=None,
                input_size_spec=None):
        r"""
        Predict samples in the spirit of `scikit-learn`.

        * It is YOUR responsibility to prepare the data in an iterable object
          of numpy arrays with the correctly matching first dimension (i.e.,
          the number of channels).
        * The method will match the data to the input size of the network and
          forward propagate it in batches.

        :param input_sequence: iterable(3D numpy arrays) or dict(string: ...).
          The 3D numpy arrays must match in their first dimension with the
          second dimension of the network input (number of channels). E.g.,
          for a network with input shape [10, 3, 24, 24], you could provide
          inputs as 4D numpy array of shape [100, 3, 10, 10] or a list of
          3D numpy arrays of shape [3, Y, X], where X and Y may be arbitrary
          and different for each image. If you are using a multi-input network,
          you can provide a dictionary of iterables of 3D numpy arrays before,
          where the keys are the input blob names to fill. Use the
          `input_processing_flags` to specify how preprocessing is
          done in any scenario.

        :param pre_batch_callbacks: list(callable) or None.
          Each element of this list will be called with the batch id as
          argument before forward propagating a batch.

        :param post_batch_callbacks: list(callable) or None.
          See before, but after a callback.

        :param out_blob_names: list(string) or None.
          The names of the blobs of which the values are returned. If
          unspecified, uses ``self.outputs``.

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

        :param input_processing_flags: dict(string:string) or None.
          A list or tuple of letters specifying the proprocessing for each
          input. 'n': no preprocessing, 'r': rescale, 'p': pad.
          Default: ['n'] * number_of_inputs

        :param output_processing_flags: dict(string:string) or None.
          A list or tuple of letters specifying the postprocessing of the
          outputs. 'n': no postprocessing, 'pX': unpack from
          padding of input X, where X is the 0-based input index.
          Default: ['n'] * number_of_inputs

        :param static_inputs: list(string) or None.
          A list of input blob names for wich the input will always be
          fully used, independent of the selected samples.

        :param input_size_spec: (int, int) or None.
          For exotic network configurations it can be necessary to specify
          the batch size and input size because it can not be automatically
          inferred. By default (None), use the first dimension of the first
          network blob as batch size and the first dimension of the associated
          inputs as amount of samples.

        :returns : dict(string:np.array) or np.array.
          Returns a dictionary of arrays if multiple outputs are returned
          or directly the array in the case of just one output.
        """
        # Parameter checks.
        if not use_fit_network and self._predict_variant is not None:
            _LOGGER.info("Using prediction network variant.")
            prednet = self._predict_variant
        else:
            prednet = self
        if isinstance(input_sequence, dict):
            # We have multiple input dicts specified.
            for input_blob_name in list(input_sequence.keys()):
                assert input_blob_name in list(prednet.blobs.keys()), (
                    "Specified input blob not found: {}.".format(
                        input_blob_name)
                )
        else:
            assert len(prednet.inputs) == 1, (
                "You did not specify the `input_sequence` as dictionary. "
                "This only works for networks with one input! Inputs: {}."
                .format(prednet.inputs)
            )
            input_sequence = {prednet.inputs[0]: input_sequence}
        if pre_batch_callbacks is None:
            pre_batch_callbacks = []
        if post_batch_callbacks is None:
            post_batch_callbacks = []
        if out_blob_names is None:
            out_blob_names = prednet.outputs
        for bname in out_blob_names:
            assert bname in prednet.blobs, (
                "The following blob name was specified in `out_blob_names` "
                "but I didn't find a blob with this name: {}.".format(bname)
            )
        # `use_fit_network` has already been checked.
        if oversample:
            assert len(prednet.inputs) == 1, (
                "Oversampling can currently only be used with 1-input networks."
            )
        input_image_dims_0 = _np.array(
            prednet.blobs[prednet.inputs[0]].data.shape[2:])
        if before_oversample_resize_to is not None:
            assert oversample is not None, (
                "`before_oversample_resize_to` specified, but not `oversample`!"
            )
            assert (before_oversample_resize_to[0] >= input_image_dims_0[0] and
                    before_oversample_resize_to[1] >= input_image_dims_0[1])
        assert input_processing_flags is None or \
            all([flag.lower() in ['r', 'p', 'n']
                 for flag in list(input_processing_flags.values())]), \
            "The input processing flags must be in ['r', 'p', 'n']."
        if input_processing_flags is None:
            input_processing_flags = dict(
                [(inp_name, 'n') for inp_name in prednet.inputs])
        assert output_processing_flags is None or \
            all([flag.lower()[0] in ['r', 'p', 'n']
                 for flag in list(output_processing_flags.values())]), \
            "The output processing flags must be in ['p', 'n']."
        if output_processing_flags is None:
            output_processing_flags = dict(
                [(out_name, 'n') for out_name in out_blob_names])
        if static_inputs is None:
            static_inputs = []
        covered_inputs = list(set(static_inputs +
                                  list(input_processing_flags.keys())))
        assert prednet.inputs == covered_inputs,\
            ("The `static_inputs` together with the "
             "`input_processing_flags` must cover all net inputs! "
             "Currently covered: {} of {}."
             .format(covered_inputs, prednet.inputs)
            )
        if input_size_spec is None:
            batch_size = prednet.blobs[prednet.inputs[0]].data.shape[0]
            nsamples = len(input_sequence[prednet.inputs[0]])
        else:
            batch_size, nsamples = input_size_spec  # pylint: disable=W0633
        if oversample:
            assert batch_size % 10 == 0,\
                ("If you use the `oversample` argument as `True`, the batch " +
                 "size must be a multiple of 10!")
            assert input_processing_flags[prednet.inputs[0]] == 'n', (
                'Automatic oversampling is only available for "n" as '
                'preprocessing.')
        ########################################################################
        # Plausibility checks done.
        # Start working!
        ########################################################################
        output_images = {output: [] for output in out_blob_names}  # pylint: disable=W0612
        cbparams = dict()
        max_iter = (nsamples if not oversample
                    else 10 * nsamples)
        cbparams['max_iter'] = max_iter
        cbparams['batch_size'] = batch_size
        cbparams['iter'] = 0
        cbparams['net'] = prednet
        cbparams['X'] = None
        chunk_size = (batch_size if not oversample else batch_size / 10)
        for chunk_idx, sample_ids in enumerate(_chunks(list(range(nsamples)),
                                                       chunk_size)):
            for inp_name in prednet.inputs:
                if inp_name not in static_inputs:
                    im_chunk = [input_sequence[inp_name][im_idx]
                                for im_idx in sample_ids]
                else:
                    # Static input.
                    im_chunk = input_sequence[inp_name]
                _LOGGER.info('Preparing chunk %d...', chunk_idx)
                for idx, im in enumerate(im_chunk):
                    if oversample:
                        if before_oversample_resize_to is not None:
                            oversampling_prep = _cv2resize(
                                _np.transpose(im, (1, 2, 0)),
                                (before_oversample_resize_to[1],
                                 before_oversample_resize_to[0]),
                                interpolation=_cv2INTER_LINEAR)
                        else:
                            oversampling_prep = _np.transpose(im, (1, 2, 0))
                        prednet.blobs[inp_name].data[idx * 10:(idx+1) * 10] =\
                            _np.transpose(
                                _caffe.io.oversample([oversampling_prep],
                                                     input_image_dims_0[:2]),
                                (0, 3, 1, 2))
                    else:
                        if inp_name in static_inputs:
                            if chunk_idx == 0:
                                # Only copy once.
                                prednet.blobs[inp_name].data[...] = im_chunk
                            continue
                        if input_processing_flags[inp_name].lower() == 'p':
                            prednet.blobs[inp_name].data[idx] = _pad(
                                im,
                                prednet.blobs[inp_name].data.shape[2:4])
                        elif input_processing_flags[inp_name].lower() == 'r':
                            prednet.blobs[inp_name].data[idx] = _np.transpose(
                                _cv2resize(
                                    _np.transpose(im, (1, 2, 0)),
                                    (prednet.blobs[inp_name].data.shape[3],
                                     prednet.blobs[inp_name].data.shape[2]),
                                    interpolation=_cv2INTER_CUBIC),
                                (2, 0, 1))
                        else:
                            # The only remaining case is
                            # input_processing_flags[inp_name].lower() == 'n'.
                            prednet.blobs[inp_name].data[idx, ...] = im
            ####################################################################
            # Data prepared.
            ####################################################################
            # Callbacks.
            cbparams['iter'] = (
                len(output_images[list(output_images.keys())[0]]) if not oversample
                else len(output_images[list(output_images.keys())[0]]) * 10)
            cbparams['X'] = [prednet.blobs[blobname]
                             for blobname in prednet.inputs]
            cbparams['callback_signal'] = 'pre_train_batch'
            for cb in pre_batch_callbacks:
                cb(cbparams)
            # Logging.
            to_image = (
                len(output_images[list(output_images.keys())[0]]) + batch_size if not oversample
                else len(output_images[list(output_images.keys())[0]]) + batch_size / 10)
            _LOGGER.info('Forward propagating chunk %d (image %d to %d of %d)...',  # noqa
                         chunk_idx,
                         len(output_images[list(output_images.keys())[0]]),
                         to_image,
                         nsamples)
            # Forward propagation.
            forward_prop_beginpoint = _time.time()
            # pylint: disable=W0212
            prednet._forward(0, len(prednet.layers) - 1)
            forward_prop_duration = _time.time() - forward_prop_beginpoint
            _LOGGER.info('Done in %ds.', forward_prop_duration)
            # Post processing.
            out = {out: prednet.blobs[out].data for out in out_blob_names}
            _LOGGER.info('Extracting output images...')
            # pylint: disable=W0612
            output_image_parts = dict((outname, []) for outname in out_blob_names)
            for blob_idx, blob_name in enumerate(out.keys()):
                full_output_image_blob = out[blob_name]
                if full_output_image_blob.ndim < 1:
                    # Scalar processing.
                    output_image_parts[blob_name].append(full_output_image_blob)
                    continue
                if output_processing_flags[blob_name].lower().startswith('p'):
                    target_output_height = \
                        prednet.blobs[
                            prednet.inputs[
                                int(output_processing_flags[
                                    blob_name][1:])]].data.shape[2]
                    target_output_width = \
                        prednet.blobs[
                            prednet.inputs[
                                int(output_processing_flags[
                                    blob_name][1:])]].data.shape[3]
                    input_image_dims = _np.array([target_output_height,
                                                  target_output_width])
                else:
                    target_output_height = None
                    target_output_width = None
                output_dims = _np.array(full_output_image_blob.shape)
                output_image_dims = output_dims[2:]
                if len(output_dims) > 2 and 'input_image_dims' in list(locals().keys()):
                    scale_h, scale_w = (input_image_dims.astype('float') /
                                        output_image_dims.astype('float'))
                else:
                    scale_h = scale_w = 1.0
                if scale_h == 1. and scale_w == 1.:
                    _LOGGER.info('No scaling necessary.')
                    scaling = False
                else:
                    _LOGGER.info('Scale change by %f, %f (h, w).',
                                 scale_h, scale_w)
                    scaling = True
                oversampled = []
                for outim_idx, outim in enumerate(full_output_image_blob):
                    if ((len(output_images[list(output_images.keys())[0]]) +\
                         len(output_image_parts[blob_name]) ==
                         nsamples and not oversample) or
                        # pylint: disable=C0330
                        ((len(output_images[list(output_images.keys())[0]]) + \
                          len(oversampled) - nsamples) <
                         (batch_size / 10) and
                         len(oversampled) % 10 == 0 and
                         len(oversampled) / 10 == \
                         (len(output_images[list(output_images.keys())[0]]) + \
                         len(oversampled) - nsamples))):
                        # The last blob was not completely used.
                        break
                    if scaling:
                        outim_resized = _np.empty((output_dims[1],
                                                   target_output_height,
                                                   target_output_width))
                        for layer_idx, layer in enumerate(outim):
                            outim_resized[layer_idx] = \
                                _cv2resize(layer,
                                           (int(target_output_width),
                                            int(target_output_height)),
                                           interpolation=_cv2INTER_CUBIC)
                    else:
                        outim_resized = outim
                    if output_processing_flags[blob_name].lower().startswith('p') \
                       and outim_resized.ndim == 3:  # Otherwise it's not clear...
                        # Padded extraction.
                        target_image_dims = _np.array(
                            input_sequence[
                                prednet.inputs[
                                    int(output_processing_flags[blob_name][1:])]]
                            [outim_idx+len(output_images[list(output_images.keys())[0]])].shape[1:],
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
                        output_image_parts[blob_name].append(extracted_work_image.copy())  # noqa
                if oversample:
                    for os_im_chunk in _chunks(oversampled, 10):
                        chunk_arr = _np.array(os_im_chunk)
                        if chunk_arr.ndim == 4:
                            assert chunk_arr.shape[0] == 10
                            # Flip back the flipped results to average correctly.
                            # The result shape is N, C, H, W, and the second half
                            # of the ten samples are horizontally flipped.
                            chunk_arr[5:] = chunk_arr[5:, :, :, ::-1]
                        output_image_parts[blob_name].append(_np.mean(chunk_arr, axis=0))
            for blob_name, part in list(output_image_parts.items()):
                output_images[blob_name].extend(part)
            cbparams['iter'] = (len(output_images[list(output_images.keys())[0]]) if not oversample
                                else len(output_images[list(output_images.keys())[0]]) * 10)
            cbparams['out'] = out
            cbparams['callback_signal'] = 'post_train_batch'
            for cb in post_batch_callbacks:
                cb(cbparams)
            del cbparams['out']
        for cb in set(pre_batch_callbacks + post_batch_callbacks):
            cb.finalize(cbparams)
        if len(output_images) == 1:
            return list(output_images.items())[0][1]
        else:
            return output_images

    def fit(self,
            iterations,
            solver,
            X=None,
            X_val=None,
            test_iterations=0,
            test_interval=0,
            test_initialization=False,
            train_callbacks=None,
            test_callbacks=None,
            read_input_batch_size_from_blob_name=None):
        """See :py:func:`barrista.solver.SolverInterface.fit`."""
        return solver.fit(iterations=iterations,
                          X=X,
                          X_val=X_val,
                          test_iterations=test_iterations,
                          test_interval=test_interval,
                          test_initialization=test_initialization,
                          train_callbacks=train_callbacks,
                          test_callbacks=test_callbacks,
                          net=self,
                          read_input_batch_size_from_blob_name=(
                              read_input_batch_size_from_blob_name))

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
