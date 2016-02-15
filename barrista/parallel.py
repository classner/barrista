# -*- coding: utf-8
"""Collection of parallel tools."""
# pylint: disable=invalid-name, protected-access, redefined-outer-name
from __future__ import print_function

import warnings as _warnings
import multiprocessing as _multiprocessing
import multiprocessing.sharedctypes as _mpsharedctypes
import numpy as _np

import barrista.monitoring as _monitoring


class DummyNet(object):  # pylint: disable=too-few-public-methods

    """Drop-in replacement to simulate blobs for the ParallelMonitors."""

    def __init__(self):
        self.blobs = {}


class DummyBlob(object):

    """Replacement network blob using SharedMemory for the ParallelMonitors."""

    def __init__(self, shared_shape, shared_data):
        self.shared_data = shared_data
        self.shared_shape = shared_shape

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
            return data[:shape[0], :shape[1], :shape[2], :shape[3]]
        else:
            return data

    def reshape(self, num, chan, height, width):
        """Simulate the blob reshape method."""
        shape = self.shape
        if len(shape) != 4:
            raise Exception("Can only reshape 4D blobs!")
        assert num == shape[0]
        assert chan == shape[1]
        realdata = _np.ctypeslib.as_array(self.shared_data)
        assert height <= realdata.shape[2]
        assert width <= realdata.shape[3]
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


def prepare_batch(cbparams):
    """Prepare a batch in an initialized filler thread."""
    # pylint: disable=global-variable-undefined, global-variable-not-assigned
    global net, cbs, train_mode, initialized, logger
    logger.debug("Preparing batch. cbparams: %s.", cbparams)
    if train_mode:
        cbparams['net'] = net
    else:
        cbparams['testnet'] = net
    if not initialized:
        cbsignal = cbparams['callback_signal']
        if train_mode:
            cbparams['callback_signal'] = 'pre_fit'
        else:
            cbparams['callback_signal'] = 'pre_test'
        for cb in cbs:
            cb(cbparams)
        cbparams['callback_signal'] = cbsignal
    for cb in cbs:
        logger.debug(type(cb))
        cb(cbparams)
        logger.debug('Done.')


def init_prebatch(self,  # pylint: disable=too-many-locals
                  net,
                  callbacks,
                  train_mode):
    """
    Initialize parallel pre-batch processing.

    Should be used with the `run_prebatch` method from this module.

    The object must have the properties:

    * _parallel_batch_res (None)
    * _no_batch_prepared (True)
    * _train_net_dummy (None)
    * _parallel_train_filler (None)
    * _test_net_dummy (None)
    * _parallel_test_filler (None)

    whereas the properties with train or test in their name are only used if
    the method is used for the respective `train_mode`.
    """
    assert self._parallel_batch_res is None
    assert self._no_batch_prepared
    if train_mode:
        assert self._train_net_dummy is None
        assert self._parallel_train_filler is None
    else:
        assert self._test_net_dummy is None
        assert self._parallel_test_filler is None
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
            tmp = _np.ctypeslib.as_ctypes(_np.zeros((bsh[0],
                                                     bsh[1],
                                                     bsh[2] * 3,
                                                     bsh[3] * 3),
                                                    dtype='float32'))
        else:
            tmp = _np.ctypeslib.as_ctypes(_np.zeros(bsh, dtype='float32'))
        tmp_sh = _np.ctypeslib.as_ctypes(_np.zeros((len(bsh)),
                                                   dtype='int'))
        shared_arr = _mpsharedctypes.Array(
            tmp._type_,
            tmp,
            lock=False)
        shared_sh = _mpsharedctypes.Array(
            tmp_sh._type_,
            tmp_sh,
            lock=False)
        dummynet.blobs[bname] = DummyBlob(shared_sh, shared_arr)
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

def run_prebatch(self,
                 callbacks,
                 cbparams,
                 train_mode,
                 iter_p1):
    """Run the prebatch callbacks."""
    if self._parallel_batch_res is not None:
        self._parallel_batch_res.get()
    if self._no_batch_prepared:
        # Run once serially.
        for cb in callbacks:
            cb(cbparams)
        self._no_batch_prepared = False
    else:
        # Copy over the prepared data.
        if train_mode:
            dummy = self._train_net_dummy
            net = cbparams['net']
        else:
            dummy = self._test_net_dummy
            net = cbparams['testnet']
        for bname in list(dummy.blobs.keys()):
            if not _np.all(dummy.blobs[bname].data.shape ==
                           net.blobs[bname].data.shape):
                net.blobs[bname].reshape(dummy.blobs[bname].data.shape)
            net.blobs[bname].data[...] = dummy.blobs[bname].data
    # Start next parallel run.
    ncbparams = {
        'iter': iter_p1,
        'callback_signal': cbparams['callback_signal'],
        'max_iter': cbparams['max_iter'],
        'batch_size': cbparams['batch_size']
    }
    if 'test_interval' in list(cbparams.keys()):
        ncbparams['test_interval'] = cbparams['test_interval']
    if train_mode:
        self._parallel_batch_res = self._parallel_train_filler.apply_async(
            prepare_batch, args=(ncbparams,))
    else:
        self._parallel_batch_res = self._parallel_test_filler.apply_async(
            prepare_batch, args=(ncbparams,))
    # Execute the serially-to-execute monitors.
    for cb in callbacks:
        if not isinstance(cb, _monitoring.ParallelMonitor):
            cb(cbparams)

def finalize_prebatch(self):
    """Cleanup workers and artifacts."""
    if hasattr(self, '_parallel_train_filler'):
        self._parallel_train_filler.terminate()
        self._parallel_train_filler.join()
        self._parallel_train_filler = None
        self._train_net_dummy = None
    if (hasattr(self, '_parallel_test_filler') and
            self._parallel_test_filler is not None):
        self._parallel_test_filler.terminate()
        self._parallel_test_filler.join()
        self._parallel_test_filler = None
        self._test_net_dummy = None
    self._parallel_batch_res = None
    self._no_batch_prepared = True
