# -*- coding: utf-8 -*-
"""Exposes the caffe solvers."""
# pylint: disable=E1101, F0401, C0103, R0913, R0914, W0212, E1121, E0611, W0406
# pylint: disable=duplicate-code, too-many-lines
from __future__ import print_function

from . import monitoring as _monitoring
from . import parallel as _parallel
# CAREFUL! This must be imported pre any caffe-related import!
from .tools import pbufToPyEnum as _pbufToPyEnum

import time as _time
import logging as _logging
import hashlib
import copy
from tempfile import NamedTemporaryFile as _NamedTemporaryFile
import numpy as _np
import google.protobuf.text_format as _gprototext

import caffe as _caffe
import caffe.proto.caffe_pb2 as _caffe_pb2

#: Describes the type of the solver used. All solver types supported by caffe
#: are available.
SolverType = _pbufToPyEnum(_caffe_pb2.SolverParameter.SolverType)

#: Describes the Phase used. All solver types supported by caffe
#: are available.
_Phase = _pbufToPyEnum(_caffe_pb2.Phase)

_HAS_ITER_SIZE = hasattr(_caffe_pb2.SolverParameter, 'iter_size')
try:
    _ADAM_SOLVER_CLASS = _caffe.AdamSolver
    _ADAM_SOLVER_ENUM = SolverType.ADAM
except AttributeError:  # pragma: no cover
    _ADAM_SOLVER_CLASS = None
    _ADAM_SOLVER_ENUM = None
try:
    _ADADELTA_SOLVER_CLASS = _caffe.AdaDeltaSolver
    _ADADELTA_SOLVER_ENUM = SolverType.ADADELTA
except AttributeError:  # pragma: no cover
    _ADADELTA_SOLVER_CLASS = None
    _ADADELTA_SOLVER_ENUM = None
try:
    _ADAGRAD_SOLVER_CLASS = _caffe.AdaGradSolver
    _ADAGRAD_SOLVER_ENUM = SolverType.ADAGRAD
except AttributeError:  # pragma: no cover
    _ADAGRAD_SOLVER_CLASS = None
    _ADAGRAD_SOLVER_ENUM = None
try:
    _RMSPROP_SOLVER_CLASS = _caffe.RMSPropSolver
    _RMSPROP_SOLVER_ENUM = SolverType.RMSPROP
except AttributeError:  # pragma: no cover
    _RMSPROP_SOLVER_CLASS = None
    _RMSPROP_SOLVER_ENUM = None

_LOGGER = _logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class Solver(object):

    """Describes the Solver concept."""

    _solver_types = {}
    _caffe_solver_type = None
    _solver_type = None

    def __init__(self, **kwargs):
        r"""
        Constructor.

        :param iter_size: int>0.
          The number of batches the gradient is accumulated over (not
          available in older caffe versions).

        :param lr_policy: string in ['fixed', 'step', ...]
          The policy to use to adjust the learning rate during fitting.
          Taken from ``solver.cpp``:

          * fixed: always return base_lr.
          * step: return base_lr \* gamma ^ (floor(iter / step))
          * exp: return base_lr \* gamma ^ iter
          * inv: return base_lr \* (1 + gamma \* iter) ^ (- power)
          * multistep: similar to step but it allows non uniform steps defined
            by stepvalue
          * poly: the effective learning rate follows a polynomial decay, to be
            zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
          * sigmoid: the effective learning rate follows a sigmod decay
            return base_lr ( 1/(1 + exp(-gamma \* (iter - stepsize))))

        :param base_lr: float or None.
          The base learning rate to use.

        :param gamma: float or None.

        :param power: float or None.

        :param weight_decay: float or None.
          Use weight decay to reduce the weights at each step.

        :param regularization_type: string in ['L1', 'L2'].
          Specifies how the ``weight_decay`` is applied.

        :param step_stepsize: float or None.
          The stepsize for the step policy.

        :param stepvalue: list(int) or None.
          The stepvalue parameter for the multistep policy.

        :param clip_gradients: float or None.
          Clips the gradients to the specified value.

        :param random_seed: int>0 or None.
          If specified, seeds the solver for reproducible results. Otherwise,
          it uses a time dependent seed.

        :param snapshot_prefix: string or None.
          If the ``Checkpointer`` monitor is used, this prefix is used to
          create the snapshots.

        :param debug_info: bool.
          If set to ``True``, gives additional output in the logs.
        """
        self._net = None
        self._parameter_hash = None
        self._parameter_dict = dict()
        self.update_parameters(**kwargs)
        # some default internal parameters
        self._parameter_dict['snapshot_after_train'] = False
        self._parameter_dict['solver_type'] = self._caffe_solver_type
        # every solver can append its on assertions or overwrite the given ones
        self._asserts = []
        if _HAS_ITER_SIZE:
            self._asserts.append(self.Assert_iter_size)
        self._asserts.append(self.Assert_regularization_types)
        self._asserts.append(self.Assert_policy)
        self._solver = None
        self._print_warning = False
        self._train_net_dummy = None
        self._test_net_dummy = None
        self._parallel_train_filler = None
        self._parallel_test_filler = None
        self._parallel_batch_res_train = None
        self._parallel_batch_res_test = None

    def restore(self, filename, net=None):
        """Restore the solverstate from a file."""
        if self._net is None:
            assert net is not None, ('you must specify a net on which the '
                                     'restored solver will be used!')
        if net is not None:
            # The method self._Update_net must not be used here, since it
            # is allowed to use a new net.
            self._net = net
        self._Update_solver()
        self._solver.restore(filename)

    @classmethod
    def Get_required_arguments(cls):
        """The minimum number of required parameters."""
        return ['base_lr']

    @classmethod
    def Get_optional_arguments(cls):
        """
        Get the optional parameters.

        Optional parameters and some of which are None
        not all combinations are possible, this is enforced by various
        asserts when calling Get_parameter_dict().
        """
        ret_dict = {'debug_info': False,
                    'weight_decay': None,
                    'lr_policy': 'fixed',
                    'regularization_type': 'L2',
                    'power': None,
                    'gamma': None,
                    'snapshot_prefix': None,
                    'stepsize': None,
                    'stepvalue': None,
                    'clip_gradients': None,
                    'random_seed': None,
                    'net': None}
        if _HAS_ITER_SIZE:
            ret_dict['iter_size'] = 1
        return ret_dict

    def fit(self,  # pylint: disable=too-many-statements, too-many-branches
            iterations,
            X=None,
            X_val=None,
            input_processing_flags=None,
            test_iterations=0,
            test_interval=0,
            test_initialization=False,
            train_callbacks=None,
            test_callbacks=None,
            net=None,
            read_input_batch_size_from_blob_name=None,
            use_fit_phase_for_validation=False,
            allow_test_phase_for_train=False,
            shuffle=False):
        r"""
        fit the network to specific data.

        Use monitors from the module :py:mod:`barrista.monitoring` as
        callbacks to monitor the state of the net and create checkpoints.
        This method offers the following kwargs to monitors (* indicates,
        that the values are only available at test time, - indicates, that
        the value is not necessarily available):

        * max_iter,
        * iter,
        * batch_size,
        * net,
        * testnet\[only if there is a test phase, i.e., X_val is set]
        * solver,
        * callback_signal\[is automatically set by the fit function],
        * X\-[only if provided by the user],
        * X_val\-[only if provided by the user],
        * [the following fields are only set if the corresponding
           loss/accuracy layer exists for the train and/or test phase.
           It can also be set by providing a custom ResultExtractor]
        * loss\-,
        * test_loss\*,
        * accuracy\-,
        * test_accuracy\*-,

        :param iterations: int.
          The number of training iterations to do. This is the plain number
          of iterations, completely disregarding the batch size, i.e., for
          ``iterations`` being 10 and ``batch_size`` being 10, just one batch
          is forward propagated.

        :param X: dict of numpy.ndarray or None.
          If specified, is used as input data. It is used sequentially, so
          shuffle it pre, if required. The keys of the dict have to have
          a corresponding layer name in the net.

        :param X_val: dict of numpy.ndarray or None.
          If specified and ``test_interval>0``, it is used as input data.
          It is used sequentially, so shuffle it pre, if required. The
          keys of the dict have to have a corresponding layer name in
          the net.

        :param input_processing_flags: dict(string, string) or None.
          See ``CyclingDataMonitor.__init__`` for the ``input_processing_flags``
          parameter. In short, if you specify your sample via list, you may
          specify for each blob, whether they should be padded 'p', or
          resized 'r' to match the network input size. If they fit perfectly,
          you may specify 'n' or omit the parameter and use ``None``.

        :param test_iterations: int.
          The number of test iterations to determine the validation score,
          if ``test_interval>0``.

        :param test_interval: int.
          The number of iterations between runs on the validation set. Is
          specified in plain iterations, disregarding batch size. Hence, it
          must be a multiple of the batch size.

        :param test_initialization: bool.
          Whether to do a run on the validation set pre the training is
          started to get an initial score.

        :param train_callbacks: list(barrista.monitoring.Monitor).
          List of callback callables. Will be called pre and post training
          batch is processed. This list will be processed
          sequentially, meaning that monitors in the sequence can
          provide information for later monitors as done with
          ResultExtractor.

        :param test_callbacks: list(callable).
          List of callback callables. Will be called for pre and post
          testing and pre and post each batch of testing processed.
          This list will be processed sequentially, meaning that
          monitors in the sequence can provide information for later
          monitors as done with ResultExtractor.

        :param read_input_batch_size_from_blob_name: string.
          The name of the layer to take the input batch size from (as the
          first dimension of its first blob). Must be specified if the
          network does not have explicit inputs (e.g., when trained from
          an LMDB).

        :param use_fit_phase_for_validation: bool.
          If set to True, do not change the phase of the net for running
          a validation step during training. This can be helpful to reduce
          memory consumption. This ignores the TEST phase of the net completely,
          but it's not necessary to use it if the data is provided by the
          Python layers.

        :param allow_test_phase_for_train: bool.
          If set to True, allow using a network in its TEST phase to be trained.
          May make sense in exotic settings, but should prevent bugs. If not
          set to True, an AssertionError is raised in this scenario.
          Why is this so important? The ``DropoutLayer`` and ``PoolLayer`` (in
          the case of stochastic pooling) are sensitive to this parameter and
          results are very different for the two settings.

        :param shuffle: bool.
          If set to True, shuffle the training data every epoch. The test data
          is not shuffled. Default: False.
        """
        if net is not None:
            from barrista import net as _net
            assert isinstance(net, _net.Net), (
                'net must be an instance of barrista.net.Net')
        self._Update_net(net)

        assert self._net is not None, (
            'neither the solver was initialized with a net nor',
            'the fit function was called with one')

        assert self._net._mode == _Phase.TRAIN or allow_test_phase_for_train, (
            'The network must be in TRAIN phase for fitting! If you really '
            'want to, you can override this requirement by setting '
            'the optional parameter `allow_test_phase_for_train` to True.'
        )

        train_callbacks = self._Assert_callbacks(self._net,
                                                 train_callbacks,
                                                 'train')

        testnet = self._Init_testnet(test_interval,
                                     use_fit_phase_for_validation)
        if testnet is not None:
            test_callbacks = self._Assert_callbacks(testnet,
                                                    test_callbacks,
                                                    'test')


        batch_size, test_iterations = self._Get_batch_size(
            self._net,
            testnet,
            test_interval,
            test_iterations,
            X_val,
            read_input_batch_size_from_blob_name)

        self._Assert_iterations(
            batch_size,
            iterations,
            test_interval,
            test_iterations,
            self._parameter_dict.get('stepvalue')
        )

        if self._parameter_dict.get('stepvalue') is not None:
            self._parameter_dict['stepvalue'] = [
                val / batch_size for val in self._parameter_dict['stepvalue']]

        self._Init_cycling_monitor(X,
                                   X_val,
                                   input_processing_flags,
                                   batch_size,
                                   test_interval,
                                   train_callbacks,
                                   test_callbacks,
                                   shuffle)
        run_pre = True
        iteration = 0
        cbparams = dict()
        cbparams['max_iter'] = iterations
        cbparams['batch_size'] = batch_size
        cbparams['iter'] = 0
        cbparams['net'] = self._net
        cbparams['testnet'] = testnet
        cbparams['solver'] = self
        cbparams['X'] = X
        cbparams['X_val'] = X_val
        cbparams['test_iterations'] = test_iterations
        cbparams['test_interval'] = test_interval
        cbparams['train_callbacks'] = train_callbacks
        cbparams['test_callbacks'] = test_callbacks
        cbparams['callback_signal'] = 'initialize_train'
        for cb in train_callbacks:
            cb(cbparams)
        if test_interval > 0:
            cbparams['callback_signal'] = 'initialize_test'
            for cb in test_callbacks:
                cb(cbparams)
        try:
            _parallel.init_prebatch(
                self,
                self._net,
                train_callbacks,
                True)
            if test_interval > 0:
                _parallel.init_prebatch(
                    self,
                    testnet,
                    test_callbacks,
                    False)
            while iteration <= iterations:
                cbparams['iter'] = iteration
                # Check whether to test the net.
                if ((  # pylint: disable=too-many-boolean-expressions
                        test_interval > 0 and
                        iteration % test_interval == 0 and iteration > 0
                    ) or (
                        iteration == 0 and test_initialization
                    ) or (
                        test_interval > 0 and iteration + batch_size > iterations
                    )
                   ):
                    ###############################################################
                    # testing loop
                    ###############################################################
                    test_iter = 0
                    run_pre = True
                    # Pretest gets called if necessary in `run_prebatch`.
                    while test_iter < test_iterations:
                        cbparams['callback_signal'] = 'pre_test_batch'
                        _parallel.run_prebatch(
                            self,
                            test_callbacks,
                            cbparams,
                            False,
                            cbparams['iter'],
                            run_pre)

                        # pylint: disable=W0212
                        testnet._forward(0, len(testnet.layers) - 1)

                        cbparams['callback_signal'] = 'post_test_batch'
                        for cb in test_callbacks:
                            cb(cbparams)
                        test_iter += batch_size
                        run_pre = False

                    cbparams['callback_signal'] = 'post_test'
                    for cb in test_callbacks:
                        cb(cbparams)
                    run_pre = True

                if iteration == iterations:
                    break

                ###################################################################
                # training loop
                ###################################################################

                # `pre_fit` gets called if necessary in `run_prebatch`.
                PRETRBATCH_BEGINPOINT = _time.time()
                cbparams['callback_signal'] = 'pre_train_batch'
                _parallel.run_prebatch(
                    self,
                    train_callbacks,
                    cbparams,
                    True,
                    cbparams['iter'] + batch_size,
                    run_pre)
                run_pre = False
                PRETRBATCH_DURATION = _time.time() - PRETRBATCH_BEGINPOINT
                _LOGGER.debug("Pre-batch preparation time: %03.3fs.",
                              PRETRBATCH_DURATION)

                TRBATCH_BEGINPOINT = _time.time()
                self.step(1)
                TRBATCH_DURATION = _time.time() - TRBATCH_BEGINPOINT
                _LOGGER.debug("Batch processing time: %03.3fs.",
                              TRBATCH_DURATION)

                POSTTRBATCH_BEGINPOINT = _time.time()
                cbparams['callback_signal'] = 'post_train_batch'
                for cb in train_callbacks:
                    cb(cbparams)
                POSTTRBATCH_DURATION = _time.time() - POSTTRBATCH_BEGINPOINT
                _LOGGER.debug("Post-batch processing time: %03.3fs.",
                              POSTTRBATCH_DURATION)

                iteration += batch_size

        finally:
            for cb in set(train_callbacks + test_callbacks):
                if not isinstance(cb, _monitoring.ParallelMonitor):
                    cb.finalize(cbparams)
            _parallel.finalize_prebatch(self, cbparams)
            if self._parameter_dict.get('stepvalue') is not None:
                self._parameter_dict['stepvalue'] = [
                    val * batch_size for val in self._parameter_dict['stepvalue']]

    def step(self, number_of_batches):
        """Run ``number_of_batches`` solver steps."""
        tmp_hash = self.Get_parameter_hash(self.Get_parameter_dict())
        if self._parameter_hash != tmp_hash:
            if self._print_warning:  # pragma: no cover
                _LOGGER.warn('WARNING: ---------------------------------------------')
                _LOGGER.warn('you are re-initializing a new solver which will delete')
                _LOGGER.warn('the weight history of the solver.')
                _LOGGER.warn('Only use this option if you know what you are doing!')
                self._print_warning = False
            self._Update_solver()
        return self._solver.step(number_of_batches)

    def Get_parameter_dict(self):
        """Get the solver describing parameters in a dictionary."""
        # work our stack of assertions followed by a weak copy of the dict
        for Tmp_assert in self._asserts:
            assert Tmp_assert()
        return copy.copy(self._parameter_dict)

    def Assert_iter_size(self):
        """Enforce the parameter constraints."""
        return self._parameter_dict['iter_size'] > 0

    def Assert_regularization_types(self):
        """Enforce the parameter constraints."""
        return self._parameter_dict['regularization_type'] in ['L1', 'L2']

    def Assert_policy(self):  # pylint: disable=R0911
        """Enforce the parameter constraints."""
        # although redundant this allows to have a quick check
        # of what is really required without loading the actuall net which
        # might take a bit of time
        if self._parameter_dict['lr_policy'] == 'fixed':
            return 'base_lr' in self._parameter_dict
        if self._parameter_dict['lr_policy'] == 'step':
            return 'gamma' in self._parameter_dict
        if self._parameter_dict['lr_policy'] == 'exp':
            return 'gamma' in self._parameter_dict
        if self._parameter_dict['lr_policy'] == 'inv':
            return ('gamma' in self._parameter_dict and
                    'power' in self._parameter_dict)
        if self._parameter_dict['lr_policy'] == 'multistep':
            return ('stepvalue' in self._parameter_dict and
                    'base_lr' in self._parameter_dict and
                    'gamma' in self._parameter_dict)
        if self._parameter_dict['lr_policy'] == 'poly':
            return 'power' in self._parameter_dict
        if self._parameter_dict['lr_policy'] == 'sigmoid':
            return 'stepsize' in self._parameter_dict
        return False

    @classmethod
    def Get_parameter_hash(cls, solver_parameter_dict):
        """Get a has of the parameter dict."""
        hash_obj = hashlib.md5()
        for key in sorted(solver_parameter_dict.keys()):
            hash_obj.update(str(key).encode('utf-8'))
            hash_obj.update(str(solver_parameter_dict[key]).encode('utf-8'))
        return str(hash_obj.hexdigest())

    @classmethod
    def Get_caffe_solver_instance(cls, solver_parameter_dict, net):
        """Get a caffe solver object."""
        # now we actually create a instance of the solver
        solver_message = _caffe_pb2.SolverParameter(**solver_parameter_dict)
        messagestr = _gprototext.MessageToString(solver_message)
        with _NamedTemporaryFile(mode='w+b', suffix='.prototxt') as tmpfile:
            tmpfile.write(bytes(messagestr.encode('utf-8')))
            tmpfile.flush()
            try:
                # Newer version of caffe with full solver init support.
                return cls.Get_caffe_solver_class(
                    solver_parameter_dict['solver_type'])._caffe_solver_class(
                        tmpfile.name, net, _caffe._caffe.NetVec(), True)
            except TypeError:
                # Fallback for older, patched versions.
                return cls.Get_caffe_solver_class(
                    solver_parameter_dict['solver_type'])._caffe_solver_class(
                        tmpfile.name, net)
        raise Exception('could not initialize solver class')

    @classmethod
    def Get_solver_class(cls, solver_type):
        """Get the solver class as string."""
        return cls._solver_types[solver_type]

    @classmethod
    def Get_caffe_solver_class(cls, caffe_solver_type):
        """Get the solver class as ``caffe_solver_type``."""
        return cls._solver_types[caffe_solver_type]

    @classmethod
    def Register_solver(cls, solver_class):
        """Register a solver class."""
        assert issubclass(solver_class, Solver)
        if solver_class._solver_type in cls._solver_types:
            raise Exception(
                ' '.join('solver',
                         solver_class._solver_type,
                         'already defined'))
        if solver_class._caffe_solver_type in cls._solver_types:
            raise Exception(
                ' '.join('solver',
                         solver_class._solver_type,
                         'already defined'))
        # we register with both access types
        cls._solver_types[solver_class._caffe_solver_type] = solver_class
        cls._solver_types[solver_class._solver_type] = solver_class

    def _Update_solver(self):
        """Re-initialize the solver."""
        # we (re-)initialize the solver
        self._solver = self.Get_caffe_solver_instance(
            self.Get_parameter_dict(),
            self._net)
        self._parameter_hash = self.Get_parameter_hash(
            self.Get_parameter_dict())

        # we only want to see the warning once
        self._print_warning = True

    def update_parameters(self, **kwargs):
        """Update the solver parameters."""
        # adding the default keys if they are not yet set
        for argument, default in list(self.Get_optional_arguments().items()):
            if argument not in self._parameter_dict and default is not None:
                self._parameter_dict[argument] = default

        # first add all parameters which are actually required
        for arg_key, arg_value in list(kwargs.items()):
            if arg_key in self.Get_required_arguments():
                self._parameter_dict[arg_key] = arg_value

        # make sure that all required arguments are set
        tmp_required_arguments = set(self.Get_required_arguments())
        intersection = tmp_required_arguments.intersection(set(kwargs.keys()))
        if intersection != tmp_required_arguments:
            raise Exception(' '.join(
                ['we are missing required arguments',
                 str(list(kwargs.keys())),
                 'vs',
                 str(self.Get_required_arguments())]))

        for arg_key, arg_value in list(kwargs.items()):
            # the very special case of passing the net
            # this will not be passed as a parameter to the parameter dict
            # but we will ensure that the net is always the same
            # as the one used for initialization
            if arg_key == 'net':
                self._Update_net(arg_value)
                continue

            if arg_key in list(self.Get_optional_arguments().keys()):
                self._parameter_dict[arg_key] = arg_value

        # we make sure that there is no spelling mistake in the kwargs
        total_arguments = set(self.Get_required_arguments())
        total_arguments = total_arguments.union(
            list(self.Get_optional_arguments().keys()))
        for argument in list(kwargs.keys()):
            if argument not in total_arguments:
                raise Exception(' '.join(
                    ['argument', argument, 'is not supported']))

    def _Update_net(self, net):
        """Check that the net remains the same."""
        # since the user could potentially provide two different nets to
        # the solver, which is not supported, thus we check that the net
        # has not changed
        if net is None:
            return
        if self._net is not None:
            if id(self._net) != id(net):
                raise Exception(' '.join(
                    ['a solver works only with one network',
                     'the network has to remain the same']))
        self._net = net

    def _Get_batch_size(self,  # pylint: disable=R0201
                        net,
                        testnet,
                        test_interval,
                        test_iterations,
                        X_val,
                        read_input_batch_size_from_blob_name):
        """Get the batch size and the test iterations."""
        if len(net.inputs) > 0:
            # Otherwise, a DB backend is used.
            batch_size = net.blobs[net.inputs[0]].data.shape[0]
            if testnet is not None:
                assert (testnet.blobs[net.inputs[0]].data.shape[0] ==
                        batch_size), ("Validation and fit network batch size "
                                      "must agree!")
            if (test_interval != 0 and
                    test_iterations == 0 and
                    X_val is not None):
                if isinstance(X_val, dict):
                    if len(X_val.values()[0]) % batch_size != 0:
                        _LOGGER.warn(
                            "The number of test samples is not a multiple "
                            "of the batch size. Test performance estimates "
                            "will be slightly off.")
                    test_iterations = _np.ceil(float(len(X_val.values()[0])) /
                                               float(batch_size)) * batch_size
                else:
                    if len(X_val) % batch_size != 0:
                        _LOGGER.warn(
                            "The number of test samples is not a multiple "
                            "of the batch size. Test performance estimates "
                            "will be slightly off.")
                    test_iterations = _np.ceil(float(len(X_val)) /
                                               float(batch_size)) * batch_size
            if read_input_batch_size_from_blob_name is not None:
                tmp_batch_size = net.blobs[
                    read_input_batch_size_from_blob_name].data.shape[0]
                assert (tmp_batch_size == batch_size), (
                    "The input size and the first dimension of "
                    "the blob to read the batch size from don't "
                    "match: {}, {}.".format(tmp_batch_size, batch_size))
            return batch_size, test_iterations

        # some kind of backend is used
        assert read_input_batch_size_from_blob_name is not None, (
            'no inputs thus the batch_size must be determined from a blob')
        batch_size = net.blobs[
            read_input_batch_size_from_blob_name].data.shape[0]
        return batch_size, test_iterations

    @classmethod
    def _Assert_iterations(cls,
                           batch_size,
                           iterations,
                           test_interval,
                           test_iterations,
                           multistep_stepvalue):
        """Make sure iterations follow all of our rules."""
        # namely being a multiple of the batch_size
        assert iterations % batch_size == 0, (
            'Error: iterations do not match {} {}'.format(iterations,
                                                          batch_size))
        if test_interval > 0:
            assert test_iterations > 0, (
                'Test iterations must be > 0 but is {}'.format(
                    test_iterations))
        # Set the configurable arguments.
        assert test_iterations >= 0, (
            'Test iterations must be >= 0 but is {}'.format(
                test_iterations))
        assert test_interval >= 0, (
            'Test interval must be >= 0 but is {}'.format(
                test_iterations))
        assert test_interval % batch_size == 0, (
            'The test interval must be a multiple of the batch size: {}, {}',
            test_iterations, batch_size)
        if multistep_stepvalue is not None:
            for val in multistep_stepvalue:
                assert val % batch_size == 0, (
                    "The step values must be multiples  of the batch size "
                    "(is given in sample iterations)! Is %d, batch size %d." % (
                        val, batch_size))


    @classmethod
    def _Assert_callbacks(cls, net, callbacks, phase):
        """Assert the callbacks work properly."""
        if callbacks is None:
            callbacks = []
        assert isinstance(callbacks, list), (
            'callbacks have to be in a list {} {}'.format(
                str(callbacks), type(callbacks)))
        for callback in callbacks:
            assert isinstance(callback, _monitoring.Monitor), (
                'a callback has to derive from montoring.Monitor')
        if 'loss' in list(net.blobs.keys()):
            callbacks.insert(0, _monitoring.ResultExtractor(
                phase + '_loss', 'loss'))
        if 'accuracy' in list(net.blobs.keys()):
            callbacks.insert(0, _monitoring.ResultExtractor(
                phase + '_accuracy', 'accuracy'))

        return callbacks

    @classmethod
    def _Init_cycling_monitor(cls,
                              X,
                              X_val,
                              input_processing_flags,
                              batch_size,
                              test_interval,
                              train_callbacks,
                              test_callbacks,
                              shuffle):
        """
        Convencience initialization function.

        ...such that the user can
        simply provide X, X_val dicts and we internally create
        the CyclingDataMonitors.
        """
        if X is not None:
            assert len(list(X.values())[0]) >= batch_size
            # safety measure, we do not want to have two different data
            # monitors in the same callback list
            for callback in train_callbacks:
                assert not isinstance(callback, _monitoring.DataMonitor), (
                    'if we use X we cannot use a data monitor')
            tmp_data_monitor = _monitoring.CyclingDataMonitor(
                X=X,
                input_processing_flags=input_processing_flags,
                shuffle=shuffle)
            train_callbacks.insert(0, tmp_data_monitor)

        if test_interval > 0 and X_val is not None:
            assert X_val is not None
            if X is not None:
                assert len(list(X_val.values())) == len(list(X.values()))

            # safety measure, we do not want to have two different data
            # monitors in the same callback list
            for callback in test_callbacks:
                assert not isinstance(callback, _monitoring.DataMonitor), (
                    'if we use X_val we cannot use a data monitor')
            tmp_data_monitor = _monitoring.CyclingDataMonitor(
                X=X_val,
                input_processing_flags=input_processing_flags)
            test_callbacks.insert(0, tmp_data_monitor)

    def _Init_testnet(self, test_interval, use_fit_phase_for_validation):
        """Initialize the test phase network."""
        testnet = None
        if test_interval > 0:
            if use_fit_phase_for_validation:
                testnet = self._net
            else:
                # Setup the test net.
                test_netspec = self._net._specification.copy()
                test_netspec.phase = _Phase.TEST
                test_netspec.predict_inputs = None
                test_netspec.predict_input_shapes = None
                testnet = test_netspec.instantiate()
                testnet.share_with(self._net)
        return testnet


class SGDSolver(Solver):

    r"""
    Thin wrapper for the vanilla SGD solver provided by the caffe framework.

    :param momentum: float or None.
      The momentum to use. Multiplies the former gradient with this factor
      and adds it to the gradient in the following step.
    """

    _solver_type = 'sgd'
    _caffe_solver_type = SolverType.SGD
    _caffe_solver_class = _caffe.SGDSolver

    def __init__(self, **kwargs):
        """Constructor."""
        Solver.__init__(self, **kwargs)

    @classmethod
    def Get_required_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        return Solver.Get_required_arguments()

    @classmethod
    def Get_optional_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        optional_arguments = Solver.Get_optional_arguments()
        optional_arguments['momentum'] = 0.0
        return optional_arguments


class AdagradSolver(Solver):

    r"""
    Thin wrapper for the Adagrad solver provided by the caffe framework.

    To understand how this solver works please inspect the
    cplusplus implementation in solver.cpp.

    The corresponding publication is called 'Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization' by
    John Duchi, Elad Hazan, Yoram Singer

    :param momentum: float or None.
      The momentum to use. Multiplies the former gradient with this factor
      and adds it to the gradient in the following step.
    """

    _solver_type = 'adagrad'
    _caffe_solver_type = _ADAGRAD_SOLVER_ENUM
    _caffe_solver_class = _ADAGRAD_SOLVER_CLASS

    def __init__(self, **kwargs):
        """See :py:class:`barrista.solver.Solver`."""
        Solver.__init__(self, **kwargs)

    @classmethod
    def Get_required_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        required_arguments = Solver.Get_required_arguments()
        required_arguments.append('delta')
        return required_arguments

    @classmethod
    def Get_optional_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        return Solver.Get_optional_arguments()


class NesterovSolver(Solver):

    r"""
    Thin wrapper for the Nesterov solver provided by the caffe framework.

    To understand how this solver works please inspect the
    cplusplus implementation in solver.cpp.

    :param momentum: float or None.
      The momentum to use. Multiplies the former gradient with this factor
      and adds it to the gradient in the following step.
    """

    _solver_type = 'nesterov'
    _caffe_solver_type = SolverType.NESTEROV
    _caffe_solver_class = _caffe.NesterovSolver

    def __init__(self, **kwargs):
        """See :py:class:`barrista.solver.Solver`."""
        Solver.__init__(self, **kwargs)

    @classmethod
    def Get_required_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        return Solver.Get_required_arguments()

    @classmethod
    def Get_optional_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        optional_arguments = Solver.Get_optional_arguments()
        optional_arguments['momentum'] = 0.0
        return optional_arguments


class RMSPropSolver(Solver):

    r"""
    Thin wrapper for the RMSProp solver provided by the caffe framework.

    To understand how this solver works please inspect the
    cplusplus implementation in solver.cpp.

    This solver has been discussed in a lecture given by Hinton.
    www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    :param rms_decay: float
      MeanSquare(t) = rms_decay*MeanSquare(t-1)+(1-rms_decay)*SquareGradient(t)
    :param delta: float
        numerical stability [useful choice 1E-8]
    """

    _solver_type = 'rmsprop'
    _caffe_solver_type = _RMSPROP_SOLVER_ENUM
    _caffe_solver_class = _RMSPROP_SOLVER_CLASS

    def __init__(self, **kwargs):
        """See :py:class:`barrista.solver.Solver`."""
        Solver.__init__(self, **kwargs)

    @classmethod
    def Get_required_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        required_arguments = Solver.Get_required_arguments()
        required_arguments.append('rms_decay')
        required_arguments.append('delta')
        return required_arguments

    @classmethod
    def Get_optional_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        return Solver.Get_optional_arguments()


class AdaDeltaSolver(Solver):

    r"""
    Thin wrapper for the AdaDelta solver provided by the caffe framework.

    To understand how this solver works please inspect the
    cplusplus implementation in solver.cpp.

    The corresponding arxiv paper is called 'ADADELTA: An Adaptive
    Learning Rate Method' by Matthew D. Zeiler.

    :param delta: float
        numerical stability [useful choice 1E-8]
    :param momentum: float or None.
      The momentum to use. Multiplies the former gradient with this factor
      and adds it to the gradient in the following step.
    """

    _solver_type = 'adadelta'
    _caffe_solver_type = _ADADELTA_SOLVER_ENUM
    _caffe_solver_class = _ADADELTA_SOLVER_CLASS

    def __init__(self, **kwargs):
        """See :py:class:`barrista.solver.Solver`."""
        Solver.__init__(self, **kwargs)

    @classmethod
    def Get_required_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        required_arguments = Solver.Get_required_arguments()
        required_arguments.append('momentum')
        return required_arguments

    @classmethod
    def Get_optional_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        optional_arguments = Solver.Get_optional_arguments()
        # epsilon
        optional_arguments['delta'] = 1E-8
        return optional_arguments


class AdamSolver(Solver):

    r"""
    Thin wrapper for the Adam solver provided by the caffe framework.

    To understand how this solver works please inspect the
    cplusplus implementation in solver.cpp.

    The corresponding arxiv paper is called ' Adam: A Method for
    Stochastic Optimization ' by Diederik Kingma, Jimmy Ba

    :param base_lr: float
        [useful choice 0.001]
    :param momentum: float.
        beta 1 useful default 0.9
    :param momentum2: float.
        beta 2 useful default 0.999
    :param delta: float
        numerical stability [useful choice 1E-8]
    """

    _solver_type = 'adam'
    _caffe_solver_type = _ADAM_SOLVER_ENUM
    _caffe_solver_class = _ADAM_SOLVER_CLASS

    def __init__(self, **kwargs):
        """See :py:class:`barrista.solver.Solver`."""
        Solver.__init__(self, **kwargs)

    @classmethod
    def Get_required_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        return Solver.Get_required_arguments()

    @classmethod
    def Get_optional_arguments(cls):
        """See :py:class:`barrista.solver.Solver`."""
        optional_arguments = Solver.Get_optional_arguments()
        # beta 1
        optional_arguments['momentum'] = 0.9
        # beta 2
        optional_arguments['momentum2'] = 0.999
        # epsilon
        optional_arguments['delta'] = 1E-8
        return optional_arguments


# register the locally specified solver
Solver.Register_solver(SGDSolver)
Solver.Register_solver(AdagradSolver)
Solver.Register_solver(NesterovSolver)
if _RMSPROP_SOLVER_CLASS is not None:
    Solver.Register_solver(RMSPropSolver)
if _ADADELTA_SOLVER_CLASS is not None:
    Solver.Register_solver(AdaDeltaSolver)
if _ADAM_SOLVER_CLASS is not None:
    Solver.Register_solver(AdamSolver)


Get_solver_class = Solver.Get_solver_class
Get_caffe_solver_class = Solver.Get_caffe_solver_class
