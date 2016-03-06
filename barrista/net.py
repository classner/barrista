# -*- coding: utf-8 -*-
# pylint: disable=E1101, F0401, R0912, R0913, R0914, C0103, duplicate-code
"""Implements an extended Net object."""


import time as _time
import logging as _logging
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

import numpy as _np
from sklearn.feature_extraction.image import extract_patches as _extract_patches

from .tools import chunks as _chunks, pbufToPyEnum as _pbufToPyEnum
import barrista.monitoring as _monitoring
import barrista.parallel as _parallel

# CAREFUL! This must be imported before any caffe-related import!
from .initialization import init as _init

import caffe as _caffe
import caffe.proto.caffe_pb2 as _caffe_pb2
_Phase = _pbufToPyEnum(_caffe_pb2.Phase)

_init()
_LOGGER = _logging.getLogger(__name__)
if _cv2 is None:
    _LOGGER.warn('Could not import cv2! Resizing is not available!')


#: Set CPU processing mode.
set_mode_cpu = _caffe.set_mode_cpu
#: Set GPU processing mode.
set_mode_gpu = _caffe.set_mode_gpu
#: Set id of GPU device.
set_device = _caffe.set_device

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
                 mode,
                 specification=None):
        """See class documentation."""
        _caffe.Net.__init__(self, specification_filename, mode)
        self._specification = specification
        self._predict_variant = None
        self._mode = mode
        if self._specification is not None:
            if (self._specification.predict_inputs is not None and
                    self._specification.predict_input_shapes is not None):
                self._predict_variant = self._specification.\
                    get_predict_net_specification().instantiate()
                # Use the blobs of this net.
                self._predict_variant.share_with(self)
        self._parallel_batch_res_test = None
        self._parallel_test_filler = None
        self._test_net_dummy = None

    def reshape_blob(self, name, *args):
        """
        Reshape a blob for train and predict net variants.

        Convenience method to reshape a blob for all network variants in use.
        Calls the caffe ``blob.reshape`` method internally. The blob must exist
        in both networks. To change a blob shape only in one variant, use
        ``net.blobs[name].reshape`` or
        ``net._predict_variant.blobs[name].reshape`` directly.

        :param name: string.
            The name of the blob to reshape.

        :param *args: int.
            The shape-defining sizes.
        """
        self.blobs[name].reshape(*args)
        if self._predict_variant is not None:
            self._predict_variant.blobs[name].reshape(*args)

    def load_blobs_from(self, filename):
        """
        Load the network weights from the specified file.

        :param filename: string.
          The filename of the file to load.
        """
        if hasattr(_caffe.Net, 'load_blobs_from'):
            # Older barrista/caffe version.
            #self.__class__.__bases__[0].load_blobs_from(self, filename)
            # _caffe.Net.load_blobs_from(self, filename)
            _caffe.Net.copy_from(self, filename)
        else:
            self.copy_from(filename)

    # pylint: disable=too-many-statements
    def predict_sliding_window(self,
                               input_sequence,
                               test_callbacks=None,
                               out_blob_names=None,
                               use_fit_network=False,
                               oversample=False,
                               extraction_step=(1, 1),
                               account_for_step=True,
                               interpolation_method=_cv2INTER_NEAREST,
                               pad_border=True,
                               overlap_combine_max=True):
        """
        Get predictions for all images in a sliding window manner.

        Similar to the :py:func:`barrista.net.Net.predict` method. See there
        for the parameter descriptions. For this method, every image must be
        at least as big as the input size. It is then sampled using sliding
        window, and for each output layer the reassembled images are returned.

        The output of the network for one patch must either be of shape
        (num_layers X 1 X 1) or of shape (num_layers X input_height X input_width)
        and currently only one output layer is supported. If the output is of the
        same shape as the input, `account_for_step` does not play a role, and the
        inputs are combined by using the max of the predictions per position.

        :param input_sequence: iterable(3D numpy arrays)
          The 3D numpy arrays must match in their first dimension with the
          second dimension of the network input (number of channels). E.g.,
          for a network with input shape [10, 3, 24, 24], you could provide
          inputs as 4D numpy array of shape [100, 3, 10, 10] or a list of
          3D numpy arrays of shape [3, Y, X], where X and Y may be arbitrary
          and different for each image. Multi-input networks are not yet
          supported by this method.

        :param test_callbacks: list(barrista.monitoring.Monitor) or None.
          List of callback callables. Will be called pre and post batch
          processing. This list will be processed sequentially, meaning that
          monitors in the sequence can provide information for later monitors
          as done with the ``ResultExtractor``.

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

        :param extraction_step: 2-tuple(int).
          Window step size in y and x direction.

        :param account_for_step: bool.
          If set to True, the output is resized with nearest neighbor
          interpolation to get a full-sized image.

        :param interpolation_method: int in {cv2.INTER_...}.
          The interpolation strategy used, if ``account_for_step`` is set and
          the ``extraction_step`` is not ``(1, 1)``.

        :param pad_border: bool.
          Whether to return images in the original image size, or by adding
          zero padded borders.

        :param overlap_combine_max: bool.
          If the network output size is equal to the input size and the
          stepsize smaller than the output, which operator to use to combine
          overlapping areas. Default: True.
        """
        if self._predict_variant is not None and not use_fit_network:
            _LOGGER.debug("Using prediction network variant.")
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
            _LOGGER.debug("Processing image %d...", im_id)
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
            results = prednet.predict(patches,
                                      test_callbacks=test_callbacks,
                                      out_blob_names=out_blob_names,
                                      use_fit_network=use_fit_network,
                                      oversample=oversample,
                                      before_oversample_resize_to=None)
            if results[0].size > results[0].shape[0]:
                assert (results[0].ndim == 3 and
                        _np.all(results[0].shape[1:3] == input_image_dims)), (
                            ("The output shape of the net must be "
                             "(X, 1, 1) or (X, input_height, input_width) "
                             "to be used with the `predict_sliding_window` "
                             "method. {} vs {}.").format(
                                 input_image_dims,
                                 results[0].shape[1:3]))
                out_im = _np.ones((results[0].shape[0],
                                   im.shape[1],
                                   im.shape[2]),
                                  dtype=results[0].dtype) * -1.
                curr_y = 0
                curr_x = 0
                for val in results:
                    # Write back with max.
                    roi = out_im[:,
                                 curr_y:curr_y+val.shape[1],
                                 curr_x:curr_x+val.shape[2]]
                    if overlap_combine_max:
                        out_im[:,
                               curr_y:curr_y+val.shape[1],
                               curr_x:curr_x+val.shape[2]] =\
                                    _np.maximum(roi, val)
                    else:
                        for c_idx in range(roi.shape[0]):
                            for y_idx in range(roi.shape[1]):
                                for x_idx in range(roi.shape[2]):
                                    if roi[c_idx, y_idx, x_idx] == -1:
                                        roi[c_idx, y_idx, x_idx] = \
                                            val[c_idx, y_idx, x_idx]
                                    else:
                                        roi[c_idx, y_idx, x_idx] = \
                                            (val[c_idx, y_idx, x_idx] +
                                             roi[c_idx, y_idx, x_idx]) / 2.
                    # Find the position in the original image.
                    if (curr_x + extraction_step[1] + input_image_dims[1]
                            > out_im.shape[2]):
                        curr_y += extraction_step[0]
                        curr_x = 0
                    else:
                        curr_x += extraction_step[1]
                output_images.append(out_im)
            else:
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
                                ("The output shape of the net must be "
                                 "(X, 1, 1) or (X, input_height, input_width) "
                                 "to be used with the `predict_sliding_window` "
                                 "method. It is {}.").format(val.shape))
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
            _LOGGER.debug("Processed image %d in %03.2fs.",
                          im_id,
                          _time.time() - image_beginpoint)
        return output_images

    def predict(self,  # pylint: disable=R0915
                input_sequence,
                test_callbacks=None,
                out_blob_names=None,
                use_fit_network=None,
                oversample=False,
                before_oversample_resize_to=None,
                input_processing_flags=None,
                output_processing_flags=None,
                static_inputs=None,
                input_size_spec=None,
                allow_train_phase_for_test=False,
                net_input_size_adjustment_multiple_of=0):
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
        :param test_callbacks: list(barrista.monitoring.Monitor) or None.
          List of callback callables. Will be called pre and post batch
          processing. This list will be processed sequentially, meaning that
          monitors in the sequence can provide information for later monitors
          as done with the ``ResultExtractor``.
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
          A list or tuple of letters specifying the preprocessing for each
          input. 'n': no preprocessing, 'rc': rescale cubic, 'rn': rescale
          nearest, 'rl': rescale linear, 'pX': pad with value X.
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
        :param allow_train_phase_for_test: bool.
          If set to True, allow using a network in its TRAIN phase for
          prediction.
          May make sense in exotic settings, but should prevent bugs. If not
          set to True, an AssertionError is raised in this scenario.
          Why is this so important? The ``DropoutLayer`` and ``PoolLayer`` (in
          the case of stochastic pooling) are sensitive to this parameter and
          results are very different for the two settings.
        :param net_input_size_adjustment_multiple_of: int.
          If set to a value>0, the networks input is resized in multiples of
          this value to take in the input images.
        """
        # Parameter checks.
        if not use_fit_network and self._predict_variant is not None:
            _LOGGER.debug("Using prediction network variant.")
            prednet = self._predict_variant
        else:
            prednet = self
        # pylint: disable=W0212
        assert prednet._mode == _Phase.TEST or\
          allow_train_phase_for_test, (
              'The net must be in TEST phase for prediction or the optional '
              'parameter `allow_train_phase_for_test` must be set to override!'
          )
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
        if test_callbacks is None:
            test_callbacks = []
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
            all([flag.lower() in ['rc', 'rn', 'rl', 'n'] or
                 flag.startswith('p') and flag[1:].isdigit()
                 for flag in list(input_processing_flags.values())]), (
                     "The input processing flags must be in ['rc', 'rn', "
                     "'rl', 'pX', 'n'] where X is the padding value.")
        if input_processing_flags is None:
            input_processing_flags = dict(
                [(inp_name, 'n') for inp_name in prednet.inputs])
        assert output_processing_flags is None or \
            all([flag.lower()[0] in ['r', 'p', 'n']
                 for flag in list(output_processing_flags.values())]), \
            "The output processing flags must be in ['r', 'p', 'n']."
        if output_processing_flags is None:
            output_processing_flags = dict(
                [(out_name, 'n') for out_name in out_blob_names])
        else:
            for blob_name in out_blob_names:
                assert blob_name in output_processing_flags.keys(), (
                    'If you specify `output_processing_flags`, you have to '
                    'specify them for every blob in `out_blob_names`! The '
                    'flag for blob {} was not found, only for {}.'
                ).format(blob_name, output_processing_flags.keys())
        if static_inputs is None:
            static_inputs = []
        covered_inputs = list(set(static_inputs +
                                  list(input_processing_flags.keys())))
        assert sorted(prednet.inputs) == sorted(covered_inputs),\
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
            assert net_input_size_adjustment_multiple_of <= 0, (
                "If oversample is set to True, the network size can not be "
                "automatically adjusted, currently.")

        self._Init_data_monitors(
            input_sequence,
            input_processing_flags,
            static_inputs,
            oversample,
            before_oversample_resize_to,
            batch_size,
            test_callbacks,
            net_input_size_adjustment_multiple_of)

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
        cbparams['testnet'] = prednet
        cbparams['X'] = None

        cbparams['callback_signal'] = 'initialize_test'
        for cb in test_callbacks:
            cb(cbparams)
        try:  # pylint: disable=too-many-nested-blocks
            _parallel.init_prebatch(self,
                                    prednet,
                                    test_callbacks,
                                    False)
            run_pre = True
            chunk_size = (batch_size if not oversample else batch_size // 10)
            for chunk_idx, _ in enumerate(_chunks(list(range(nsamples)),
                                                  chunk_size)):
                _LOGGER.debug('Preparing chunk %d...', chunk_idx)
                # Callbacks.
                cbparams['iter'] = (
                    len(output_images[list(output_images.keys())[0]]) if not oversample
                    else len(output_images[list(output_images.keys())[0]]) * 10)
                iter_p1 = (cbparams['iter'] + batch_size if not oversample else
                           cbparams['iter'] + batch_size // 10)
                # `pre_test` gets called automatically in `run_prebatch`.
                cbparams['callback_signal'] = 'pre_test_batch'
                prebatch_beginpoint = _time.time()
                _parallel.run_prebatch(self,
                                       test_callbacks,
                                       cbparams,
                                       False,
                                       iter_p1,
                                       run_pre)
                prebatch_duration = _time.time() - prebatch_beginpoint
                _LOGGER.debug('Pre-batch preparation time: %03.3fs.',
                              prebatch_duration)
                run_pre = False

                # Logging.
                to_image = (
                    len(output_images[list(output_images.keys())[0]]) + batch_size if not oversample
                    else len(output_images[list(output_images.keys())[0]]) + batch_size / 10)
                _LOGGER.debug('Forward propagating chunk %d (image %d to %d of %d)...',  # noqa
                              chunk_idx,
                              len(output_images[list(output_images.keys())[0]]),
                              to_image,
                              nsamples)

                # Forward propagation.
                forward_prop_beginpoint = _time.time()
                prednet._forward(0, len(prednet.layers) - 1)  # pylint: disable=W0212
                forward_prop_duration = _time.time() - forward_prop_beginpoint
                _LOGGER.debug('Done in %03.3fs.', forward_prop_duration)
                # Post processing.
                out = {out: prednet.blobs[out].data for out in out_blob_names}
                _LOGGER.debug('Extracting output images...')
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
                        _LOGGER.debug('No scaling necessary.')
                        scaling = False
                    else:
                        _LOGGER.debug('Scale change by %f, %f (h, w).',
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
                                [outim_idx+
                                 len(output_images[list(output_images.keys())[0]])].shape[1:],
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
                            output_image_parts[blob_name].append(
                                extracted_work_image.copy())
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
                cbparams['iter'] = (len(output_images[list(output_images.keys())[0]])
                                    if not oversample
                                    else len(output_images[list(output_images.keys())[0]]) * 10)
                cbparams['out'] = out
                cbparams['callback_signal'] = 'post_test_batch'
                for cb in test_callbacks:
                    cb(cbparams)
                del cbparams['out']
        finally:
            for cb in test_callbacks:
                if not isinstance(cb, _monitoring.ParallelMonitor):
                    cb.finalize(cbparams)
            _parallel.finalize_prebatch(self, cbparams)
        if len(output_images) == 1:
            return list(output_images.items())[0][1]
        else:
            return output_images

    @classmethod
    def _Init_data_monitors(cls,
                            X,
                            input_processing_flags,
                            static_inputs,
                            oversample,
                            before_oversample_resize_to,
                            batch_size,
                            test_callbacks,
                            net_input_size_adjustment_multiple_of):
        """
        Convencience initialization function.

        ...such that the user can
        simply provide X dict and we internally create
        the CyclingDataMonitor.
        """
        data_prov_blobs = [key for (key, val) in X.items() if val is not None]
        if len(data_prov_blobs) > 0:
            if len(static_inputs) > 0:
                static_data_monitor = _monitoring.StaticDataMonitor(
                    dict(item for item in data_prov_blobs
                         if item in static_inputs))
                test_callbacks.insert(0, static_data_monitor)
            if oversample:
                if before_oversample_resize_to is not None:
                    os_data_monitor = _monitoring.OversamplingDataMonitor(
                        dict((name, 'l') for name in data_prov_blobs
                             if name not in static_inputs),
                        dict((name, before_oversample_resize_to)
                             for name in data_prov_blobs
                             if name not in static_inputs))
                else:
                    os_data_monitor = _monitoring.OversamplingDataMonitor(
                        dict((name, None) for name in data_prov_blobs if
                             name not in static_inputs), None)
                test_callbacks.insert(0, os_data_monitor)
                ccl_data_monitor = _monitoring.CyclingDataMonitor(
                    X=dict(item for item in X.items()
                           if item[0] not in static_inputs and
                           item[0] in data_prov_blobs),
                    only_preload=[item for item in data_prov_blobs
                                  if item not in static_inputs],
                    virtual_batch_size=batch_size//10)
                test_callbacks.insert(0, ccl_data_monitor)
            else:
                blobinfos = {}
                plain_inp_proc_flags = {}
                for key, val in input_processing_flags.items():
                    if val.startswith('p'):
                        # Go for the resizing monitor.
                        blobinfos[key] = int(val[1:])
                    else:
                        plain_inp_proc_flags[key] = val
                if len(blobinfos) > 0:
                    res_data_monitor = _monitoring.ResizingMonitor(
                        blobinfos=blobinfos,
                        base_scale=1.,
                        random_change_up_to=0.,
                        net_input_size_adjustment_multiple_of=\
                        net_input_size_adjustment_multiple_of)
                    test_callbacks.insert(0, res_data_monitor)
                    ccl_data_monitor = _monitoring.CyclingDataMonitor(
                        X=dict(item for item in X.items()
                               if item[0] not in static_inputs and
                               item[0] in list(blobinfos.keys())),
                        only_preload=[item for item in data_prov_blobs
                                      if item in list(blobinfos.keys())])
                    test_callbacks.insert(0, ccl_data_monitor)
                if len(plain_inp_proc_flags) > 0:
                    ccl_data_monitor = _monitoring.CyclingDataMonitor(
                        X=dict(item for item in X.items()
                               if item[0] not in static_inputs and
                               item[0] in list(plain_inp_proc_flags.keys())),
                        input_processing_flags=plain_inp_proc_flags)
                    test_callbacks.insert(0, ccl_data_monitor)

    def fit(self,
            iterations,
            solver,
            X=None,
            X_val=None,
            input_processing_flags=None,
            test_iterations=0,
            test_interval=0,
            test_initialization=False,
            train_callbacks=None,
            test_callbacks=None,
            read_input_batch_size_from_blob_name=None,
            use_fit_phase_for_validation=False,
            allow_test_phase_for_train=False):
        """See :py:func:`barrista.solver.SolverInterface.fit`."""
        return solver.fit(iterations=iterations,
                          X=X,
                          X_val=X_val,
                          input_processing_flags=input_processing_flags,
                          test_iterations=test_iterations,
                          test_interval=test_interval,
                          test_initialization=test_initialization,
                          train_callbacks=train_callbacks,
                          test_callbacks=test_callbacks,
                          net=self,
                          read_input_batch_size_from_blob_name=(
                              read_input_batch_size_from_blob_name),
                          use_fit_phase_for_validation=(
                              use_fit_phase_for_validation
                          ),
                          allow_test_phase_for_train=allow_test_phase_for_train)

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
