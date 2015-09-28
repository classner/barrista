# -*- coding: utf-8 -*-
"""Defines several tools for monitoring net activity."""
import logging as _logging
import os as _os
_LOGGER = _logging.getLogger(__name__)


class Monitor(object):

    """
    The monitor interface.

    Should be implemented by any monitor class. The method
    :py:func:`barrista.monitoring.Monitor.__call__` must be specified,
    the function :py:func:`barrista.monitoring.Monitor.finalize` may
    optionally be specified.
    """

    def __call__(self, **kwargs):
        """
        The call implementation.

        For available keyword arguments, see the documentation of
        :py:class:`barrista.net.Net.fit`.
        """
        raise NotImplementedError()

    def finalize(self, **kwargs):
        """Will be called at the end of a training/fitting process."""
        pass


class _LossIndicator(object):

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


class ProgressIndicator(Monitor):

    r"""
    Generates a progress bar with current information about the process.

    The progress bar always displays completion percentag and ETA. If
    available, it also displays loss, accuracy, test loss and test accuracy.

    It makes use of the following keyword arguments (\* indicates required):

    * ``iter``\*,
    * ``max_iter``\*,
    * ``loss``,
    * ``test_loss``,
    * ``accuracy``,
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

    def __call__(self, **kwargs):
        """Callback implementation."""
        if self.pbar is None:
            if 'loss' in kwargs.keys():
                widgets = [_LossIndicator(self)] + self.widgets
            else:
                widgets = self.widgets
            self.pbar = self.pbarclass(maxval=kwargs['max_iter'],
                                       widgets=widgets)
            self.pbar.start()
        if 'loss' in kwargs.keys():
            self.loss = kwargs['loss']
        if 'test_loss' in kwargs.keys():
            self.test_loss = kwargs['test_loss']
        if 'accuracy' in kwargs.keys():
            self.accuracy = kwargs['accuracy']
        if 'test_accuracy' in kwargs.keys():
            self.test_accuracy = kwargs['test_accuracy']
        self.pbar.update(value=kwargs['iter'])

    def finalize(self, **kwargs):  # pylint: disable=W0613
        """Call ``progressbar.finish()``."""
        if self.pbar is not None:
            self.pbar.finish()


class JSONLogger(Monitor):

    r"""
    Logs available information to a JSON file.

    The information is stored in a dictionary of lists. The lists contain
    score information and the iteration at which it was obtained. The
    currently logged scores are loss, accuracy, test loss and test accuracy.

    The logger makes use of the following keyword arguments
    (\* indicates required):

    * ``iter``\*,
    * ``loss``,
    * ``test_loss``,
    * ``accuracy``,
    * ``test_accuracy``.

    :param path: string.
      The path to store the file in.

    :param name: string.
      The filename. Will be prefixed with `barrista_` and `.json` will be
      appended.
    """

    def __init__(self, path, name):
        """See class documentation."""
        import json
        self.json_package = json
        self.json_filename = _os.path.join(path, 'barrista_' + name + '.json')
        self.losses = []
        self.loss_iters = []
        self.accys = []
        self.accys_iters = []
        self.test_losses = []
        self.test_loss_iters = []
        self.test_accys = []
        self.test_accy_iters = []
        self.dict = {'train': [], 'test': [], 'barrista_produced': True}

    def __call__(self, **kwargs):
        """Callback implementation."""
        if 'loss' in kwargs:
            self.losses.append(kwargs['loss'])
            self.loss_iters.append(kwargs['iter'])
            self.dict['train'].append({'NumIters': kwargs['iter'],
                                       'loss': kwargs['loss']})
        if 'accuracy' in kwargs:
            self.accys.append(kwargs['accuracy'])
            self.accys_iters.append(kwargs['iter'])
            self.dict['train'].append({'NumIters': kwargs['iter'],
                                       'accuracy': kwargs['accuracy']})
        if 'test_loss' in kwargs:
            self.test_losses.append(kwargs['test_loss'])
            self.test_loss_iters.append(kwargs['iter'])
            self.dict['test'].append({'NumIters': kwargs['iter'],
                                      'loss': kwargs['test_loss']})
        if 'test_accuracy' in kwargs:
            self.test_accys.append(kwargs['test_accuracy'])
            self.test_accy_iters.append(kwargs['iter'])
            self.dict['test'].append({'NumIters': kwargs['iter'],
                                      'accuracy': kwargs['test_accuracy']})
        with open(self.json_filename, 'w') as outf:
            self.json_package.dump(self.dict, outf)

    def finalize(self, **kwargs):  # pylint: disable=W0613
        """Write the json file."""
        with open(self.json_filename, 'w') as outf:
            self.json_package.dump(self.dict, outf)


class Checkpointer(Monitor):

    r"""
    Writes the network blobs to disk at certain iteration intervals.

    The logger makes use of the following keyword arguments
    (\* indicates required):

    * ``iter``\*,
    * ``net``\*,
    * ``batch_size``\*.

    :param name_prefix: string.
      The first part of the output filenames to generate. The current
      iteration is added, as well as '.caffemodel'.

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

    def __call__(self, **kwargs):
        """Callback implementation."""
        if not self.iterations % kwargs['batch_size'] == 0:
            print('monitoring',self.iterations,kwargs['batch_size'])
            import sys
            sys.exit(1)
        if kwargs['iter'] % self.iterations == 0:
            checkpoint_filename = self.name_prefix +\
                str(kwargs['iter']) +\
                '.caffemodel'
            _LOGGER.debug("Writing checkpoint to file '%s'.", checkpoint_filename)  # noqa
            kwargs['net'].save(checkpoint_filename)

    def finalize(self, **kwargs):
        """Write a final checkpoint."""
        if kwargs['iter'] % self.iterations != 0:
            checkpoint_filename = self.name_prefix +\
                str(kwargs['iter']) +\
                '.caffemodel'
            _LOGGER.debug("Writing checkpoint to file '%s'.", checkpoint_filename)  # noqa
            kwargs['net'].save(checkpoint_filename)


class __dummy(object):
    pass
