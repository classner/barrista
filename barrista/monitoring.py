# -*- coding: utf-8 -*-
"""Defines several tools for monitoring net activity."""
# pylint: disable=F0401, E1101
import logging as _logging
import os as _os
import numpy as _np
# pylint: disable=no-name-in-module
from scipy.stats import bernoulli as _bernoulli
from scipy.ndimage.interpolation import rotate as _rotate
from .tools import pad as _pad
try:
    import cv2 as _cv2
    _cv2INTER_CUBIC = _cv2.INTER_CUBIC  # pylint: disable=invalid-name
    _cv2INTER_LINEAR = _cv2.INTER_LINEAR  # pylint: disable=invalid-name
    _cv2INTER_NEAREST = _cv2.INTER_NEAREST  # pylint: disable=invalid-name
    _cv2resize = _cv2.resize  # pylint: disable=invalid-name
except ImportError:
    _cv2 = None
    _cv2INTER_CUBIC = None  # pylint: disable=invalid-name
    _cv2INTER_LINEAR = None  # pylint: disable=invalid-name
    _cv2INTER_NEAREST = None  # pylint: disable=invalid-name
    _cv2resize = None  # pylint: disable=invalid-name
_LOGGER = _logging.getLogger(__name__)


class Monitor(object):  # pylint: disable=R0903

    """
    The monitor interface.

    Should be implemented by any monitor class. The method
    :py:func:`barrista.monitoring.Monitor.__call__` must be specified,
    the function :py:func:`barrista.monitoring.Monitor.finalize` may
    optionally be specified.
    """

    def __call__(self, kwargs):
        """
        The call implementation.

        For available keyword arguments, see the documentation of
        :py:class:`barrista.solver.SolverInterface.Fit`.
        """
        if kwargs['callback_signal'] == 'pre_fit':
            self._pre_fit(kwargs)
        elif kwargs['callback_signal'] == 'pre_test':
            self._pre_test(kwargs)
        elif kwargs['callback_signal'] == 'post_test':
            self._post_test(kwargs)
        elif kwargs['callback_signal'] == 'pre_test_batch':
            self._pre_test_batch(kwargs)
        elif kwargs['callback_signal'] == 'post_test_batch':
            self._post_test_batch(kwargs)
        elif kwargs['callback_signal'] == 'pre_train_batch':
            self._pre_train_batch(kwargs)
        elif kwargs['callback_signal'] == 'post_train_batch':
            self._post_train_batch(kwargs)

    def _pre_fit(self, kwargs):  # pylint: disable=C0111
        pass

    def _pre_test(self, kwargs):  # pylint: disable=C0111
        pass

    def _post_test(self, kwargs):  # pylint: disable=C0111
        pass

    def _pre_test_batch(self, kwargs):  # pylint: disable=C0111
        pass

    def _post_test_batch(self, kwargs):  # pylint: disable=C0111
        pass

    def _pre_train_batch(self, kwargs):  # pylint: disable=C0111
        pass

    def _post_train_batch(self, kwargs):  # pylint: disable=C0111
        pass

    def finalize(self, kwargs):
        """Will be called at the end of a training/fitting process."""
        pass


class DataMonitor(Monitor):  # pylint: disable=R0903

    r"""
    Monitor interface for filling the blobs of a network.

    This is a specific monitor which will fill the blobs of the network
    for the forward pass or solver step.
    Ideally there should only be one such monitor per callback,
    but multiple ones are indeed possible.
    """

    pass


class CyclingDataMonitor(Monitor):  # pylint: disable=R0903

    r"""
    Uses the data sequentially.

    This monitor maps data to the network an cycles through the data
    sequentially. It is the default monitor used if a user provides X
    or X_val to the barrista.sovler.fit method.

    If further processing of the original data is intended, by using the flag
    ``only_preload``, the following monitors find a dictionary of lists of
    the original datapoints with the name 'data_orig' in their ``kwargs``.
    The data is in this case NOT written to the network input layers! This
    can make sense, e.g., for the ``ResizingMonitor``.

    :param X: dict of numpy.ndarray or list,  or None.
      If specified, is used as input data. It is used sequentially, so
      shuffle it pre, if required. The keys of the dict must have
      a corresponding layer name in the net. The values must be provided
      already in network dimension order, i.e., usually channels, height,
      width.

    :param only_preload: list(string).
      List of blobs for which the data will be loaded and stored in a dict
      of (name: list) for further processing with other monitors.

    :param input_processing_flags: dict(string, string) or None.
      If the samples are specified via list, they may have to be size adjusted
      to the network. You may specify for each blob a type of preprocessing
      from 'n' (none, default, size must fit), to 'pX' (pad, where X is the
      padding value (int) to use) or 'rY' (resize), where Y in ['c', 'n']
      (Cubic or Nearest interpolation).
    """

    def __init__(self, X, only_preload=None, input_processing_flags=None):
        """See class documentation."""
        if only_preload is None:
            only_preload = []
        self.only_preload = only_preload
        self._X = X  # pylint: disable=C0103
        if input_processing_flags is None:
            input_processing_flags = dict()
        self._input_processing_flags = input_processing_flags
        for key in input_processing_flags.keys():
            assert key in self._X.keys()
        self._padvals = dict()
        for key, val in input_processing_flags.items():
            assert (val in ['n', 'rn', 'rc'] or
                    val.startswith('p'))
            if val.startswith('p'):
                self._padvals[key] = int(val[1:])
        for key in self.only_preload:
            assert key in self._X.keys()
        self._sample_pointer = 0
        self._len_data = None
        self._batch_size = None

    def _pre_fit(self, kwargs):
        # we make sure, now that the network is available, that
        # all names in the provided data dict has a corresponding match
        # in the network
        net = kwargs['net']
        if 'test' in kwargs['callback_signal']:
            net = kwargs['testnet']

        self._len_data = len(list(self._X.values())[0])
        for key, value in list(self._X.items()):
            if key not in self._input_processing_flags:
                self._input_processing_flags[key] = 'n'
            assert key in list(net.blobs.keys()), (
                'data key has no corresponding network blob {} {}'.format(
                    key, str(list(net.blobs.keys()))))
            assert len(value) == self._len_data, (
                'all items need to have the same length {} vs {}'.format(
                    len(value), self._len_data))
            assert isinstance(value, _np.ndarray) or isinstance(value, list), (
                'data must be a numpy nd array or list ({})'.format(type(value))
            )
        self._batch_size = net.blobs[list(self._X.keys())[0]].data.shape[0]

    def _pre_train_batch(self, kwargs):
        self._pre_batch(kwargs['net'], kwargs)

    def _pre_test_batch(self, kwargs):
        self._pre_batch(kwargs['testnet'], kwargs)

    def _pre_batch(self, net, kwargs):  # pylint: disable=C0111, W0613, R0912
        # this will simply cycle through the data.
        samples_ids = [idx % self._len_data for idx in
                       range(self._sample_pointer,
                             self._sample_pointer + self._batch_size)]
        # updating the sample pointer for the next time
        self._sample_pointer = (
            (self._sample_pointer + len(samples_ids)) % self._len_data)

        if len(self.only_preload) > 0:
            sample_dict = dict()

        for key in list(self._X.keys()):  # pylint: disable=too-many-nested-blocks
            if key in self.only_preload:
                sample_dict[key] = []
            # this will actually fill the data for the network
            for sample_idx in range(self._batch_size):
                if key in self.only_preload:
                    sample_dict[key].append(
                        self._X[key][samples_ids[sample_idx]])
                else:
                    if (net.blobs[key].data[sample_idx].size == 1 and (
                            isinstance(self._X[key][samples_ids[sample_idx]],
                                       (int, float)) or
                            self._X[key][samples_ids[sample_idx]].size == 1) or
                            self._X[key][samples_ids[sample_idx]].size ==
                            net.blobs[key].data[sample_idx].size):
                        if net.blobs[key].data[sample_idx].size == 1:
                            net.blobs[key].data[sample_idx] =\
                                self._X[key][samples_ids[sample_idx]]
                        else:
                            net.blobs[key].data[sample_idx] = (
                                self._X[key][samples_ids[sample_idx]].reshape(
                                    net.blobs[key].data.shape[1:]))
                    else:
                        if self._input_processing_flags[key] == 'n':
                            raise Exception(("Sample size {} does not match " +
                                             "network input size {} and no " +
                                             "preprocessing is allowed!")
                                            .format(
                                                self._X[key][samples_ids[sample_idx]].size,
                                                net.blobs[key].data[sample_idx].size))
                        elif self._input_processing_flags[key] in ['rn', 'rc']:
                            assert (
                                self._X[key][samples_ids[sample_idx]].shape[0]
                                == net.blobs[key].data.shape[1])
                            if self._input_processing_flags == 'rn':
                                interp_method = _cv2INTER_NEAREST
                            else:
                                interp_method = _cv2INTER_CUBIC
                            for channel_idx in range(
                                    net.blobs[key].data.shape[1]):
                                net.blobs[key].data[sample_idx, channel_idx] =\
                                    _cv2resize(
                                        self._X[key][samples_ids[sample_idx]]
                                        [channel_idx],
                                        (net.blobs[key].data.shape[3],
                                         net.blobs[key].data.shape[2]),
                                        interpolation=interp_method)
                        else:
                            # Padding.
                            net.blobs[key].data[sample_idx] = _pad(
                                self._X[key][samples_ids[sample_idx]],
                                net.blobs[key].data.shape[2:4],
                                val=self._padvals[key])
        if len(self.only_preload) > 0:
            kwargs['data_orig'] = sample_dict

    def finalize(self, kwargs):  # pylint: disable=W0613
        """Nothing to do here."""
        pass


class ResizingMonitor(Monitor):  # pylint: disable=R0903

    r"""
    Optionally resizes input data and adjusts the network input shape.

    This monitor optionally resizes the input data randomly and adjusts
    the network input size accordingly (this works only for batch size 1
    and fully convolutional networks).

    For this to work, it must be used with the ``CyclingDataMonitor`` with
    ``only_preload`` set.

    :param blobinfos: dict(string, int).
      Describes which blobs to apply the resizing operation to, and which
      padding value to use for the remaining space.

    :param base_scale: float.
      If set to a value different than 1., apply the given base scale first
      to images. If set to a value different than 1., the parameter
      ``interp_methods`` must be set.

    :param random_change_up_to: float.
      If set to a value different than 0., the scale change is altered
      randomly with a uniformly drawn value from -``random_change_up_to`` to
      ``random_change_up_to``, that is being added to the base value.

    :param net_input_size_adjustment_multiple_of: int.
      If set to a value greater than 0, the blobs shape is adjusted from its
      initial value (which is used as minimal one) in multiples of the given
      one.

    :param interp_methods: dict(string, string).
      Dictionary which stores for every blob the interpolation method. The
      string must be for each blob in ['n', 'c'] (nearest neighbour, cubic).
    """

    def __init__(self,  # pylint: disable=R0913
                 blobinfos,
                 base_scale=1.,
                 random_change_up_to=0.,
                 net_input_size_adjustment_multiple_of=0,
                 interp_methods=None):
        """See class documentation."""
        self._blobinfos = blobinfos
        self._base_scale = base_scale
        self._random_change_up_to = random_change_up_to
        if self._base_scale != 1. or self._random_change_up_to != 0.:
            assert interp_methods is not None
            for key in self._blobinfos.keys():
                assert key in interp_methods.keys()
                assert interp_methods[key] in ['n', 'c']
        self._interp_methods = interp_methods
        self._adjustment_multiple_of = net_input_size_adjustment_multiple_of
        self._min_input_size = None
        self._batch_size = None

    def _pre_fit(self, kwargs):
        # we make sure, now that the network is available, that
        # all names in the provided data dict have a corresponding match
        # in the network
        net = kwargs['net']
        if 'test' in kwargs['callback_signal']:
            net = kwargs['testnet']

        for key in list(self._blobinfos.keys()):
            assert key in list(net.blobs.keys()), (
                'data key has no corresponding network blob {} {}'.format(
                    key, str(list(net.blobs.keys()))))
            assert net.blobs[key].data.ndim == 4
            if self._adjustment_multiple_of > 0:
                if self._min_input_size is None:
                    self._min_input_size = net.blobs[key].data.shape[2:4]
                else:
                    assert net.blobs[key].data.shape[2:4] == self._min_input_size, (
                        'if automatic input size adjustment is activated, all inputs '
                        'must be of same size (first: {}, {}: {})'.format(
                            self._min_input_size, key, net.blobs[key].data.shape[2:4]))
        self._batch_size = net.blobs[list(self._blobinfos.keys())[0]].data.shape[0]
        if self._adjustment_multiple_of > 0:
            assert self._batch_size == 1, (
                "If size adjustment is activated, the batch size must be one!")

    def _pre_train_batch(self, kwargs):
        self._pre_batch(kwargs['net'], kwargs)

    def _pre_test_batch(self, kwargs):
        self._pre_batch(kwargs['testnet'], kwargs)

     # pylint: disable=C0111, W0613, R0912, too-many-locals
    def _pre_batch(self, net, kwargs):
        scales = None
        sizes = None
        for key, value in kwargs['data_orig'].items():
            assert len(value) == self._batch_size
            if sizes is None:
                sizes = []
                for img in value:
                    sizes.append(img.shape[1:3])
            else:
                for img_idx, img in enumerate(value):
                    # pylint: disable=unsubscriptable-object
                    assert img.shape[1:3] == sizes[img_idx]
        for key, padval in self._blobinfos.items():
            if scales is None:
                scales = []
                for sample_idx in range(self._batch_size):
                    if self._random_change_up_to > 0:
                        scales.append(
                            self._base_scale +
                            _np.random.uniform(low=-self._random_change_up_to,
                                               high=self._random_change_up_to))
                    else:
                        scales.append(self._base_scale)
            for sample_idx in range(self._batch_size):
                # Get the scaled data.
                scaled_sample = kwargs['data_orig'][key][sample_idx]
                if scales[sample_idx] != 1.:
                    scaled_sample = _np.empty((scaled_sample.shape[0],
                                               int(scaled_sample.shape[1] *
                                                   scales[sample_idx]),
                                               int(scaled_sample.shape[2] *
                                                   scales[sample_idx])),
                                              dtype='float32')
                    if self._interp_methods[key] == 'n':
                        interpolation_method = _cv2INTER_NEAREST
                    else:
                        interpolation_method = _cv2INTER_CUBIC
                    for layer_idx in range(scaled_sample.shape[0]):
                        scaled_sample[layer_idx] = _cv2resize(
                            kwargs['data_orig'][key][sample_idx][layer_idx],
                            (scaled_sample.shape[2],
                             scaled_sample.shape[1]),
                            interpolation=interpolation_method)
                # If necessary, adjust the network input size.
                if self._adjustment_multiple_of > 0:
                    image_height, image_width = scaled_sample.shape[1:3]
                    netinput_height = int(max(
                        self._min_input_size[0] +
                        _np.ceil(
                            float(image_height - self._min_input_size[0]) /
                            self._adjustment_multiple_of) *
                        self._adjustment_multiple_of,
                        self._min_input_size[0]))
                    netinput_width = int(max(
                        self._min_input_size[1] +
                        _np.ceil(
                            float(image_width - self._min_input_size[1]) /
                            self._adjustment_multiple_of) *
                        self._adjustment_multiple_of,
                        self._min_input_size[1]))
                    net.blobs[key].reshape(1,
                                           scaled_sample.shape[0],
                                           netinput_height,
                                           netinput_width)
                # Put the data in place.
                net.blobs[key].data[sample_idx] = _pad(
                    scaled_sample,
                    net.blobs[key].data.shape[2:4],
                    val=padval)

    def finalize(self, kwargs):  # pylint: disable=W0613
        """Nothing to do here."""
        pass


# pylint: disable=too-few-public-methods
class RotatingMirroringMonitor(Monitor):

    r"""
    Rotate and/or horizontally mirror samples within blobs.

    For every sample, the rotation and mirroring will be consistent
    across the blobs.

    :param blobinfos: dict(string, int).
      A dictionary containing the blob names and the padding values that
      will be applied.

    :param max_rotation_degrees: float.
      The rotation will be sampled uniformly from the interval
      [-rotation_degrees, rotation_degrees[ for each sample.

    :param mirror_prob: float.
      The probability that horizontal mirroring occurs. Is as well sampled
      individually for every sample.
    """

    def __init__(self,
                 blobinfos,
                 max_rotation_degrees,
                 mirror_prob=0.):
        """See class documentation."""
        self._blobinfos = blobinfos
        self._rotation_degrees = max_rotation_degrees
        self._mirror_prob = mirror_prob
        self._batch_size = None

    def _pre_fit(self, kwargs):
        # we make sure, now that the network is available, that
        # all names in the provided data dict have a corresponding match
        # in the network
        net = kwargs['net']
        if 'test' in kwargs['callback_signal']:
            net = kwargs['testnet']

        for key in list(self._blobinfos.keys()):
            assert key in list(net.blobs.keys()), (
                'data key has no corresponding network blob {} {}'.format(
                    key, str(list(net.blobs.keys()))))
            assert net.blobs[key].data.ndim == 4
        self._batch_size = net.blobs[list(self._blobinfos.keys())[0]].data.shape[0]

    def _pre_train_batch(self, kwargs):
        self._pre_batch(kwargs['net'], kwargs)

    def _pre_test_batch(self, kwargs):
        self._pre_batch(kwargs['testnet'], kwargs)

     # pylint: disable=C0111, W0613, R0912, too-many-locals
    def _pre_batch(self, net, kwargs):
        rotations = None
        mirrorings = None
        spline_interpolation_order = 0
        prefilter = False
        for key, padval in self._blobinfos.items():
            if rotations is None:
                rotations = []
                if self._rotation_degrees > 0.:
                    rotations = _np.random.uniform(low=-self._rotation_degrees,
                                                   high=self._rotation_degrees,
                                                   size=self._batch_size)
                else:
                    rotations = [0.] * self._batch_size
            if mirrorings is None:
                mirrorings = []
                if self._mirror_prob > 0.:
                    mirrorings = _bernoulli.rvs(self._mirror_prob,
                                                size=self._batch_size)
                else:
                    mirrorings = [0] * self._batch_size
            for sample_idx in range(self._batch_size):
                if rotations[sample_idx] != 0.:
                    net.blobs[key].data[sample_idx] = _rotate(
                        net.blobs[key].data[sample_idx],
                        rotations[sample_idx],
                        (1, 2),
                        reshape=False,
                        order=spline_interpolation_order,
                        mode='constant',
                        cval=padval,
                        prefilter=prefilter)
                if mirrorings[sample_idx] == 1.:
                    net.blobs[key].data[sample_idx] = \
                        net.blobs[key].data[sample_idx, :, :, ::-1]

    def finalize(self, kwargs):  # pylint: disable=W0613
        """Nothing to do here."""
        pass


class _LossIndicator(object):  # pylint: disable=R0903

    r"""
    A plugin indicator for the ``progressbar`` package.

    This must be used in conjunction with the
    :py:class:`barrista.monitoring.ProgressIndicator`. If available, it
    outputs current loss, accuracy, test loss and test accuracy.

    :param progress_indicator:
      :py:class:`barrista.monitoring.ProgressIndicator`. The information
      source to use.
    """

    def __init__(self, progress_indicator):
        self.progress_indicator = progress_indicator

    def __call__(self, pbar, stats):
        r"""Compatibility with new versions of ``progressbar2``."""
        return self.update(pbar)

    def update(self, pbar):  # pylint: disable=W0613
        """The update method to implement by the ``progressbar`` interface."""
        if self.progress_indicator.loss is not None:
            ret_val = 'Loss: {0:.4f}'.format(self.progress_indicator.loss)
        else:
            ret_val = 'Loss: -----'
        if self.progress_indicator.accuracy is not None:
            ret_val += '|Accy: {0:.4f}'.format(
                self.progress_indicator.accuracy)
        if self.progress_indicator.test_loss is not None:
            ret_val += '|TLoss: {0:.4f}'.format(
                self.progress_indicator.test_loss)
        if self.progress_indicator.test_accuracy is not None:
            ret_val += '|TAccy: {0:.4f}'.format(
                self.progress_indicator.test_accuracy)
        return ret_val


class ResultExtractor(Monitor):  # pylint: disable=R0903

    r"""
    This monitor is designed for monitoring scalar layer results.

    The main use case are salar outputs such as loss and accuracy.

    IMPORTANT: this monitor will change cbparams and add new values to it,
    most likely other monitors will depend on this, thus, ResultExtractors
    should be among the first monitors in the callback list, e.g. by
    insert them always in the beginning.

    It will extract the value of a layer and add the value to the cbparam.

    :param cbparam_key: string.
      The key we will overwrite/set in the cbparams dict

    """

    def __init__(self, cbparam_key, layer_name):
        """See class documentation."""
        self._layer_name = layer_name
        self._cbparam_key = cbparam_key
        self._init = False
        self._not_layer_available = True
        self._test_data = None

    def __call__(self, kwargs):
        """Callback implementation."""
        if self._not_layer_available and self._init:
            return
        Monitor.__call__(self, kwargs)

    def _pre_fit(self, kwargs):
        tmp_net = kwargs['net']
        if 'test' in kwargs['callback_signal']:
            tmp_net = kwargs['testnet']
        if self._layer_name in list(tmp_net.blobs.keys()):
            self._not_layer_available = False
        self._init = True
        assert self._cbparam_key not in kwargs, (
            'it is only allowed to add keys to the cbparam,',
            'not overwrite them {} {}'.format(self._cbparam_key,
                                              list(kwargs.keys())))

    def _pre_train_batch(self, kwargs):
        kwargs[self._cbparam_key] = 0.0

    def _post_train_batch(self, kwargs):
        kwargs[self._cbparam_key] = float(
            kwargs['net'].blobs[self._layer_name].data[...].ravel()[0])

    def _pre_test(self, kwargs):
        self._test_data = []

    def _post_test(self, kwargs):
        kwargs[self._cbparam_key] = _np.mean(self._test_data)

    def _post_test_batch(self, kwargs):
        # need to multiply by batch_size since it is normalized
        # internally
        self._test_data.append(float(
            kwargs['testnet'].blobs[self._layer_name].data[...].ravel()[0]))
        kwargs[self._cbparam_key] = self._test_data[-1]


class ProgressIndicator(Monitor):  # pylint: disable=R0903

    r"""
    Generates a progress bar with current information about the process.

    The progress bar always displays completion percentag and ETA. If
    available, it also displays loss, accuracy, test loss and test accuracy.

    It makes use of the following keyword arguments (\* indicates required):

    * ``iter``\*,
    * ``max_iter``\*,
    * ``train_loss``,
    * ``test_loss``,
    * ``train_accuracy``,
    * ``test_accuracy``.
    """

    def __init__(self):
        """See class documentation."""
        self.loss = None
        self.test_loss = None
        self.accuracy = None
        self.test_accuracy = None
        from progressbar import ETA, Percentage, Bar, ProgressBar
        self.widgets = [Bar(), Percentage(), ' ', ETA()]
        self.pbarclass = ProgressBar
        self.pbar = None

    def _post_train_batch(self, kwargs):
        if self.pbar is None:
            if 'train_loss' in list(kwargs.keys()):
                widgets = [_LossIndicator(self)] + self.widgets
            else:
                widgets = self.widgets
            self.pbar = self.pbarclass(maxval=kwargs['max_iter'],
                                       widgets=widgets)
            self.pbar.start()
        if 'train_loss' in list(kwargs.keys()):
            self.loss = kwargs['train_loss']
        if 'train_accuracy' in list(kwargs.keys()):
            self.accuracy = kwargs['train_accuracy']
        self.pbar.update(value=kwargs['iter'])

    def _post_test(self, kwargs):
        if self.pbar is None:
            if 'test_loss' in list(kwargs.keys()):
                widgets = [_LossIndicator(self)] + self.widgets
            else:
                widgets = self.widgets
            self.pbar = self.pbarclass(maxval=kwargs['max_iter'],
                                       widgets=widgets)
            self.pbar.start()
        if 'test_loss' in list(kwargs.keys()):
            self.test_loss = kwargs['test_loss']
        if 'test_accuracy' in list(kwargs.keys()):
            self.test_accuracy = kwargs['test_accuracy']
        self.pbar.update(value=kwargs['iter'])

    def finalize(self, kwargs):  # pylint: disable=W0613
        """Call ``progressbar.finish()``."""
        if self.pbar is not None:
            self.pbar.finish()


class JSONLogger(Monitor):  # pylint: disable=R0903

    r"""
    Logs available information to a JSON file.

    The information is stored in a dictionary of lists. The lists contain
    score information and the iteration at which it was obtained. The
    currently logged scores are loss, accuracy, test loss and test accuracy.

    The logger makes use of the following keyword arguments
    (\* indicates required):

    * ``iter``\*,

    :param path: string.
      The path to store the file in.

    :param name: string.
      The filename. Will be prefixed with 'barrista_' and '.json' will be
      appended.

    :param logging: dict of lists.
      The two keys in the dict which are used are test, train.
      For each of those a list of keys can be provided, those keys
      have to be available in the kwargs/cbparams structure.
      Usually the required data is provided by the ResultExtractor.
    """

    def __init__(self, path, name, logging):
        """See class documentation."""
        import json
        self.json_package = json
        self.json_filename = _os.path.join(path, 'barrista_' + name + '.json')
        self.dict = {'train': [], 'test': [], 'barrista_produced': True}
        self._logging = logging

    def __call__(self, kwargs):
        """Callback implementation."""
        Monitor.__call__(self, kwargs)
        with open(self.json_filename, 'w') as outf:
            self.json_package.dump(self.dict, outf)

    def _pre_fit(self, kwargs):
        for key in list(self._logging.keys()):
            assert key in ['train', 'test'], (
                'only train and test is supported by this logger')

    def _post_test(self, kwargs):
        self._post('test', kwargs)

    def _post_train_batch(self, kwargs):
        self._post('train', kwargs)

    def _post(self, phase_name, kwargs):  # pylint: disable=C0111
        if phase_name not in self._logging:
            return
        for key in self._logging[phase_name]:
            if key in kwargs:
                self.dict[phase_name].append({'NumIters': kwargs['iter'],
                                              key: kwargs[key]})

    def finalize(self, kwargs):  # pylint: disable=W0613
        """Write the json file."""
        with open(self.json_filename, 'w') as outf:
            self.json_package.dump(self.dict, outf)


class Checkpointer(Monitor):  # pylint: disable=R0903

    r"""
    Writes the network blobs to disk at certain iteration intervals.

    The logger makes use of the following keyword arguments
    (\* indicates required):

    * ``iter``\*,
    * ``net``\*,
    * ``batch_size``\*.

    :param name_prefix: string or None.
      The first part of the output filenames to generate. The prefix '_iter_,
      the current iteration, as well as '.caffemodel' is added.

      If you are using a caffe version from later than Dec. 2015, caffe's
      internal snapshot method is exposed to Python and also snapshots the
      solver. If it's available, then this method will be used. However,
      in that case, it's not possible to influence the storage location
      from Python. Please use the solver parameter ``snapshot_prefix``
      when constructing the solver instead (this parameter may be None
      and is unused then).

    :param iterations: int > 0.
      Always if the current number of iterations is divisible by iterations,
      the network blobs are written to disk. Hence, this value must be a
      multiple of the batch size!
    """

    def __init__(self,
                 name_prefix,
                 iterations):
        """See class documentation."""
        assert iterations > 0
        _LOGGER.info('Setting up checkpointing with name prefix %s every ' +
                     '%d iterations.', name_prefix, iterations)
        self.name_prefix = name_prefix
        self.iterations = iterations

    def _post_train_batch(self, kwargs):
        assert self.iterations % kwargs['batch_size'] == 0, (
            'iterations not multiple of batch_size, {} vs {}'.format(
                self.iterations, kwargs['batch_size']))

        if kwargs['iter'] % self.iterations == 0 and kwargs['iter'] > 0:
            # pylint: disable=protected-access
            if not hasattr(kwargs['solver']._solver, 'snapshot'):
                checkpoint_filename = (
                    self.name_prefix + '_iter_' +
                    str(int(kwargs['iter'] /
                            kwargs['batch_size']) + 1) +
                    '.caffemodel')
                _LOGGER.debug("Writing checkpoint to file '%s'.",
                              checkpoint_filename)
                kwargs['net'].save(checkpoint_filename)
            else:
                # pylint: disable=protected-access
                kwargs['solver']._solver.snapshot()
                caffe_checkpoint_filename = ('_iter_' +
                                             str(kwargs['iter'] /
                                                 kwargs['batch_size'] + 1) +
                                             '.caffemodel')
                caffe_sstate_filename = ('_iter_' +
                                         str(kwargs['iter'] /
                                             kwargs['batch_size'] + 1) +
                                         '.solverstate')
                _LOGGER.debug('Writing checkpoint to file "[solverprefix]%s" ' +
                              'and "[solverprefix]%s".',
                              caffe_checkpoint_filename,
                              caffe_sstate_filename)

    def finalize(self, kwargs):
        """Write a final checkpoint."""
        self._post_train_batch(kwargs)
