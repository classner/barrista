# -*- coding: utf-8
"""Collection of parallel tools."""
# pylint: disable=invalid-name, protected-access, redefined-outer-name
from __future__ import print_function

import warnings as _warnings
import multiprocessing as _multiprocessing
import numpy as _np

import barrista.monitoring as _monitoring


class DummyNet(object):  # pylint: disable=too-few-public-methods

    """Drop-in replacement to simulate blobs for the ParallelMonitors."""

    def __init__(self):
        self._filled = False
        self.blobs = {}


class DummyBlob(object):

    """Replacement network blob using SharedMemory for the ParallelMonitors."""

    def __init__(self, shared_shape, shared_data, real_shape):
        self.shared_data = shared_data
        self.shared_shape = shared_shape
        self.real_shape = real_shape

    @property
    def shape(self):
        """Get the current blob shape."""
        shape = _np.ctypeslib.as_array(self.shared_shape)
        return shape

    @property
    def data(self):
        """Get the blob data."""
        data = _np.ctypeslib.as_array(self.shared_data)
        shape = self.shape
        if len(shape) == 4:
            return data.reshape(self.real_shape)[
                :shape[0], :shape[1], :shape[2], :shape[3]]
        else:
            return data.reshape(self.real_shape)

    def reshape(self, num, chan, height, width):
        """Simulate the blob reshape method."""
        shape = self.shape
        if len(shape) != 4:
            raise Exception("Can only reshape 4D blobs!")
        assert num == shape[0]
        assert chan == shape[1]
        assert height <= self.real_shape[2]
        assert width <= self.real_shape[3]
        shape[2] = height
        shape[3] = width


def init_filler(dummynet, filler_cbs, in_train_mode):
    """Initialize a filler thread."""
    # pylint: disable=global-variable-undefined, global-variable-not-assigned
    global net, cbs, train_mode, initialized, logger
    logger = _multiprocessing.log_to_stderr()
    logger.debug("Initializing filler. Train mode: %s.", in_train_mode)
    net = dummynet
    cbs = filler_cbs
    train_mode = in_train_mode
    initialized = False


def run_cbs(cbparams):
    """Run the callbacks in this filler thread."""
    # pylint: disable=global-variable-undefined, global-variable-not-assigned
    global net, cbs, train_mode, initialized, logger
    logger.debug("Preparing batch. cbparams: %s.", cbparams)
    if train_mode:
        cbparams['net'] = net
    else:
        cbparams['testnet'] = net
    for cb in cbs:
        cb(cbparams)


def finalize_cbs(cbparams):
    """Finalize the callbacks in this filler thread."""
    # pylint: disable=global-variable-undefined, global-variable-not-assigned
    global cbs, logger
    logger.debug("Finalizing callbacks.")
    for cb in cbs:
        cb.finalize(cbparams)


def init_prebatch(self,  # pylint: disable=too-many-locals
                  net,
                  callbacks,
                  train_mode):
    """
    Initialize parallel pre-batch processing.

    Should be used with the `run_prebatch` method from this module.

    The object must have the properties:

    * _parallel_batch_res_train (None)
    * _parallel_batch_rest_test (None)
    * _train_net_dummy (None)
    * _parallel_train_filler (None)
    * _test_net_dummy (None)
    * _parallel_test_filler (None)

    whereas the properties with train or test in their name are only used if
    the method is used for the respective `train_mode`.
    """
    if train_mode:
        assert self._parallel_batch_res_train is None
        assert self._train_net_dummy is None
        assert self._parallel_train_filler is None
    else:
        assert self._test_net_dummy is None
        assert self._parallel_test_filler is None
        assert self._parallel_batch_res_test is None
    parallelcbs = [cb for cb in callbacks
                   if isinstance(cb, _monitoring.ParallelMonitor)]
    nublobnames = []
    for cb in parallelcbs:
        nublobnames.extend(cb.get_parallel_blob_names())
    dummyblobs = list(set(nublobnames))
    dummyshapes = [list(net.blobs[db].shape) for db in dummyblobs]
    dummynet = DummyNet()
    for bname, bsh in zip(dummyblobs, dummyshapes):
        if len(bsh) == 4:
            real_shape = (bsh[0], bsh[1], bsh[2] * 3, bsh[3] * 3)
        else:
            real_shape = bsh
        shared_arr = _multiprocessing.Array(
            'f',
            _np.zeros(_np.prod(real_shape), dtype='float32'),
            lock=False)
        shared_sh = _multiprocessing.Array(
            'i',
            _np.zeros(len(bsh), dtype='int'),
            lock=False)
        dummynet.blobs[bname] = DummyBlob(shared_sh, shared_arr, real_shape)
        with _warnings.catch_warnings():
            # For more information on why this is necessary, see
            # https://www.reddit.com/r/Python/comments/j3qjb/parformatlabpool_replacement
            _warnings.simplefilter('ignore', RuntimeWarning)
            dummynet.blobs[bname].shape[...] = bsh
    filler_cbs = [cb for cb in callbacks
                  if isinstance(cb, _monitoring.ParallelMonitor)]
    if train_mode:
        self._train_net_dummy = dummynet
        self._parallel_train_filler = _multiprocessing.Pool(
            1,
            initializer=init_filler,
            initargs=(dummynet, filler_cbs, True))
    else:
        self._test_net_dummy = dummynet
        self._parallel_test_filler = _multiprocessing.Pool(
            1,
            initializer=init_filler,
            initargs=(dummynet, filler_cbs, False))


def _extract_ncbparams(cbparams):
    ncbparams = {
        'iter': cbparams['iter'],
        'callback_signal': cbparams['callback_signal'],
        'max_iter': cbparams['max_iter'],
        'batch_size': cbparams['batch_size']
    }
    if 'test_interval' in list(cbparams.keys()):
        ncbparams['test_interval'] = cbparams['test_interval']
    return ncbparams


def run_prebatch(self,  # pylint: disable=too-many-branches, too-many-arguments
                 callbacks,
                 cbparams,
                 train_mode,
                 iter_p1,
                 run_pre):
    """Run the prebatch callbacks."""
    # Prepare the parameters for the parallel workers.
    ncbparams = _extract_ncbparams(cbparams)
    if train_mode:
        dummy = self._train_net_dummy
        net = cbparams['net']
    else:
        dummy = self._test_net_dummy
        net = cbparams['testnet']
    if run_pre:
        # Run pre_test or pre_fit.
        callback_signal = 'pre_fit' if train_mode else 'pre_test'
        cbs_orig = cbparams['callback_signal']
        cbparams['callback_signal'] = callback_signal
        for cb in [cb for cb in callbacks
                   if not isinstance(cb, _monitoring.ParallelMonitor)]:
            cb(cbparams)
        cbparams['callback_signal'] = cbs_orig
        # For the parallel workers.
        ncbparams['callback_signal'] = callback_signal
        if train_mode:
            self._parallel_train_filler.apply(run_cbs,
                                              args=(ncbparams,))
        else:
            self._parallel_test_filler.apply(run_cbs,
                                             args=(ncbparams,))
            # Set the test dummy as cleared, so to achieve reproducibility
            # for test results if the test dataset size is not a multiple of
            # batch size times test iterations.
            dummy._filled = False
        ncbparams['callback_signal'] = cbs_orig
    if not dummy._filled:
        # Run the callbacks.
        for cb in [callb for callb in callbacks
                   if not isinstance(callb, _monitoring.ParallelMonitor)]:
            cb(cbparams)
        if train_mode:
            self._parallel_batch_res_train =\
                self._parallel_train_filler.apply_async(
                    run_cbs,
                    args=(ncbparams,))
        else:
            self._parallel_batch_res_test =\
                self._parallel_test_filler.apply_async(
                    run_cbs,
                    args=(ncbparams,))
        dummy._filled = True
    # Get the parallel results.
    if train_mode:
        self._parallel_batch_res_train.get()
    else:
        self._parallel_batch_res_test.get()
    # Copy over the prepared data.
    for bname in list(dummy.blobs.keys()):
        if not _np.all(dummy.blobs[bname].data.shape ==
                       net.blobs[bname].data.shape):
            net.blobs[bname].reshape(dummy.blobs[bname].data.shape)
        net.blobs[bname].data[...] = dummy.blobs[bname].data
    # Start next parallel run.
    ncbparams['iter'] = iter_p1
    if train_mode:
        self._parallel_batch_res_train =\
            self._parallel_train_filler.apply_async(
                run_cbs, args=(ncbparams,))
    else:
        self._parallel_batch_res_test =\
            self._parallel_test_filler.apply_async(
                run_cbs, args=(ncbparams,))
    # Execute the serially-to-execute monitors.
    for cb in callbacks:
        if not isinstance(cb, _monitoring.ParallelMonitor):
            cb(cbparams)

def finalize_prebatch(self, cbparams):
    """Cleanup workers and artifacts."""
    ncbparams = _extract_ncbparams(cbparams)
    if hasattr(self, '_parallel_train_filler'):
        self._parallel_train_filler.apply(finalize_cbs, args=(ncbparams,))
        self._parallel_train_filler.close()
        self._parallel_train_filler.join()
        self._parallel_train_filler = None
        self._train_net_dummy = None
        self._parallel_batch_res_train = None
    if (hasattr(self, '_parallel_test_filler') and
            self._parallel_test_filler is not None):
        self._parallel_test_filler.apply(finalize_cbs, args=(ncbparams,))
        self._parallel_test_filler.close()
        self._parallel_test_filler.join()
        self._parallel_test_filler = None
        self._test_net_dummy = None
        self._parallel_batch_res_test = None
