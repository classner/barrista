# -*- coding: utf-8 -*-
"""Defines several tools for monitoring net activity."""
# pylint: disable=F0401, E1101
import logging as _logging
import os as _os
import numpy as _np
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

    :param X: dict of numpy.ndarray or None.
      If specified, is used as input data. It is used sequentially, so
      shuffle it pre, if required. The keys of the dict have to have
      a corresponding layer name in the net.
    """

    def __init__(self, **kwargs):
        """See class documentation."""
        self._X = kwargs['X']  # pylint: disable=C0103
        self._sample_pointer = 0
        self._len_data = None
        self._batch_size = None

    def _pre_fit(self, kwargs):
        # we make sure, now the the network is available that
        # all names in the provided data dict has a corresponding match
        # in the network
        net = kwargs['net']
        if 'test' in kwargs['callback_signal']:
            net = kwargs['testnet']

        self._len_data = len(list(self._X.values())[0])
        for key, value in list(self._X.items()):
            assert key in list(net.blobs.keys()), (
                'data key has no corresponding network blob {} {}'.format(
                    key, str(list(net.blobs.keys()))))
            assert len(value) == self._len_data, (
                'all items need to have the same length {} vs {}'.format(
                    len(value), self._len_data))
            assert isinstance(value, _np.ndarray), (
                'data must be a numpy nd array {}'.format(type(value)))
        self._batch_size = net.blobs[list(self._X.keys())[0]].data.shape[0]

    def _pre_train_batch(self, kwargs):
        self._pre_batch(kwargs['net'], kwargs)

    def _pre_test_batch(self, kwargs):
        self._pre_batch(kwargs['testnet'], kwargs)

    def _pre_batch(self, net, kwargs):  # pylint: disable=C0111, W0613
        # this will simply cycle through the data.
        samples_ids = [idx % self._len_data for idx in
                       range(self._sample_pointer,
                             self._sample_pointer + self._batch_size)]

        # updating the sample pointer for the next time
        self._sample_pointer = (
            (self._sample_pointer + len(samples_ids)) % self._len_data)

        for key in list(self._X.keys()):
            # this will actually fill the data for the network
            net.blobs[key].data[...] = (
                self._X[key][samples_ids].reshape(
                    net.blobs[key].data.shape))

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
                    str(kwargs['iter'] /
                        kwargs['batch_size'] + 1) +
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
