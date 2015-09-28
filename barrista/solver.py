# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""Parameter wrapper for solver."""

import hashlib
import copy
from tempfile import NamedTemporaryFile as _NamedTemporaryFile

import google.protobuf.text_format as _gprototext

# CAREFUL! This must be imported before any caffe-related import!
from tools import pbufToPyEnum as _pbufToPyEnum

import caffe as _caffe
import caffe.proto.caffe_pb2 as _caffe_pb2

#: Describes the type of the solver used. All solver types supported by caffe
#: are available.
SolverType = _pbufToPyEnum(_caffe_pb2.SolverParameter.SolverType)


class SolverInterface(object):

    _solver_types = {}
    _caffe_solver_type = None
    _solver_type = None

    r"""
        :param iter_size: int>0.
          The number of batches the gradient is accumulated over.

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

        :param stepvalue: float or None.
          The stepvalue parameter for the multistep policy.

        :param clip_gradients: float or None.
          Clips the gradients to the specified value.

        :param random_seed: int>0 or None.
          If specified, seeds the solver for reproducible results. Otherwise,
          it uses a time dependent seed.

        :param debug_info: bool.
          If set to ``True``, gives additional output in the logs.
    """
    def __init__(self, **kwargs):
        self._net = None
        self._parameter_hash = None
        self._parameter_dict = dict()

        self.Update_parameters(**kwargs)
        # some default internal parameters
        self._parameter_dict['snapshot_after_train'] = False
        self._parameter_dict['solver_type'] = self._caffe_solver_type

        # every solver can append its on assertions or overwrite the given ones
        self._asserts = []
        self._asserts.append(self.Assert_iter_size)
        self._asserts.append(self.Assert_regularization_types)
        self._asserts.append(self.Assert_policy)

        self._print_warning = False

    def _Update_solver(self):
        # we initialize the solver
        self._solver = self.Get_caffe_solver_instance(
            self.Get_parameter_dict(),
            self._net)
        self._parameter_hash = self.Get_parameter_hash(
            self.Get_parameter_dict())

        # we only want to see the warning once
        self._print_warning = True

    def Update_parameters(self, **kwargs):
        # adding the default keys if they are not yet set
        for argument, default in self.Get_optional_arguments().items():
            if argument not in self._parameter_dict and default is not None:
                self._parameter_dict[argument] = default

        # first add all parameters which are actually required
        for arg_key, arg_value in kwargs.items():
            if arg_key in self.Get_required_arguments():
                self._parameter_dict[arg_key] = arg_value

        # make sure that all required arguments are set
        tmp_required_arguments = set(self.Get_required_arguments())
        intersection = tmp_required_arguments.intersection(set(kwargs.keys()))
        if intersection != tmp_required_arguments:
            raise Exception(' '.join(
                ['we are missing required arguments',
                 str(kwargs.keys()),
                 'vs',
                 str(self.Get_required_arguments())]))

        for arg_key, arg_value in kwargs.items():
            # the very special case of passing the net
            # this will not be passed as a parameter to the parameter dict
            # but we will ensure that the net is always the same
            # as the one used for initialization
            if arg_key == 'net':
                self._Update_net(arg_value)
                    
            if arg_key in self.Get_optional_arguments().keys():
                self._parameter_dict[arg_key] = arg_value

        # we make sure that there is no spelling mistake in the kwargs
        total_arguments = set(self.Get_required_arguments())
        total_arguments = total_arguments.union(
            self.Get_optional_arguments().keys())
        for argument in kwargs.keys():
            if argument not in total_arguments:
                raise Exception(' '.join(
                    ['argument', argument, 'is not supported']))

    def _Update_net(self, net):
        if self._net is not None:
            if id(self._net) != id(net):
                raise Exception(' '.join(
                    ['a solver works only with one network',
                        'the network has to remain the same']))
        self._net = net

    def Step(self, number_of_batches, net=None):
        if net is not None:
            self._Update_net(net)
        tmp_hash = self.Get_parameter_hash(self.Get_parameter_dict())
        if self._parameter_hash != tmp_hash:
            if self._print_warning:
                print('WARNING::---------------------------------------------')
                print('you are re-initializing a new solver which will delete')
                print('the weight history of the solver.')
                print('Only use this option if you know what you are doing')
                self._print_warning = False
            self._Update_solver()
        return self._solver.step(number_of_batches)

    @classmethod
    def Get_required_arguments(cls):
        # the minimum number of required parameters
        return ['base_lr']

    @classmethod
    def Get_optional_arguments(cls):
        # optional parameters and some of which are None
        # not all combinations are possible, this is enforced by various
        # asserts when calling Get_parameter_dict()
        return {'iter_size': 1,
                'debug_info': False,
                'weight_decay': None,
                'lr_policy': 'fixed',
                'regularization_type': 'L2',
                'power': None,
                'gamma': None,
                'stepsize': None,
                'stepvalue': None,
                'clip_gradients': None,
                'random_seed': None,
                'net': None}
    
    def Get_parameter_dict(self):
        # work our stack of assertions followed by a weak copy of the dict
        for Tmp_assert in self._asserts:
            assert Tmp_assert()
        return copy.copy(self._parameter_dict)

    def Assert_iter_size(self):
        return self._parameter_dict['iter_size'] > 0

    def Assert_regularization_types(self):
        return self._parameter_dict['regularization_type'] in ['L1', 'L2']

    def Assert_policy(self):
        # this is kind of redundant but allows to have a quick check
        # of what is really required without loading the actuall net which
        # might take a bit of time
        if self._parameter_dict['lr_policy'] == 'fixed':
            return True
        if self._parameter_dict['lr_policy'] == 'step':
            return 'gamma' in self._parameter_dict
        if self._parameter_dict['lr_policy'] == 'exp':
            return 'gamma' in self._parameter_dict
        if self._parameter_dict['lr_policy'] == 'inv':
            return ('gamma' in self._parameter_dict and
                    'power' in self._parameter_dict)
        if self._parameter_dict['lr_policy'] == 'multistep':
            return ('stepvalue' in self._parameter_dict and
                    'stepsize' in self._parameter_dict and
                    'gamma')
        if self._parameter_dict['lr_policy'] == 'poly':
            return 'power' in self._parameter_dict
        if self._parameter_dict['lr_policy'] == 'sigmoid':
            return 'stepsize' in self._parameter_dict
        return False

    @classmethod
    def Get_parameter_hash(cls, solver_parameter_dict):
        hash_obj = hashlib.md5()
        for key in sorted(solver_parameter_dict.keys()):
            hash_obj.update(str(key))
            hash_obj.update(str(solver_parameter_dict[key]))
        return str(hash_obj.hexdigest())

    @classmethod
    def Get_caffe_solver_instance(cls, solver_parameter_dict, net):
        # now we actually create a instance of the solver
        solver_message = _caffe_pb2.SolverParameter(**solver_parameter_dict)
        messagestr = _gprototext.MessageToString(solver_message)
        with _NamedTemporaryFile(mode='w+b', suffix='.prototxt') as tmpfile:
            tmpfile.write(messagestr)
            tmpfile.flush()
            return cls.Get_caffe_solver_class(
                solver_parameter_dict['solver_type'])._caffe_solver_class(
                    tmpfile.name, net)
        raise Exception('could not initialize solver class')

    @classmethod
    def Get_solver_class(cls, solver_type):
        # convenience function to get the solver class as a string
        return cls._solver_types[solver_type]

    @classmethod
    def Get_caffe_solver_class(cls, caffe_solver_type):
        # convenience function to get the solver class as a caffe_solver_type
        return cls._solver_types[caffe_solver_type]

    @classmethod
    def Register_solver(cls, solver_class):
        # register different solvers
        assert issubclass(solver_class, SolverInterface)
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


class SGDSolver(SolverInterface):

    _solver_type = 'sgd'
    _caffe_solver_type = SolverType.SGD
    _caffe_solver_class = _caffe.SGDSolver

    r"""
        :param momentum: float or None.
          The momentum to use. Multiplies the former gradient with this factor
          and adds it to the gradient in the following step.
    """
    def __init__(self, **kwargs):
        SolverInterface.__init__(self, **kwargs)

    @classmethod
    def Get_required_arguments(cls):
        return SolverInterface.Get_required_arguments()

    @classmethod
    def Get_optional_arguments(cls):
        optional_arguments = SolverInterface.Get_optional_arguments()
        optional_arguments['momentum'] = 0.0
        return optional_arguments


class AdagradSolver(SolverInterface):

    _solver_type = 'adagrad'
    _caffe_solver_type = SolverType.ADAGRAD
    _caffe_solver_class = _caffe.AdaGradSolver

    r"""
        :param adagrad_delta: float or None.
          Parameter for the AdaGrad solver.
    """
    def __init__(self, **kwargs):
        SolverInterface.__init__(self, **kwargs)

    @classmethod
    def Get_required_arguments(cls):
        required_arguments = SolverInterface.Get_required_arguments()
        required_arguments.append('delta')
        return required_arguments

    @classmethod
    def Get_optional_arguments(cls):
        return SolverInterface.Get_optional_arguments()


class NesterovSolver(SolverInterface):

    _solver_type = 'nesterov'
    _caffe_solver_type = SolverType.NESTEROV
    _caffe_solver_class = _caffe.NesterovSolver

    r"""
        :param momentum: float or None.
          The momentum to use. Multiplies the former gradient with this factor
          and adds it to the gradient in the following step.
    """
    def __init__(self, **kwargs):
        SolverInterface.__init__(self, **kwargs)

    @classmethod
    def Get_required_arguments(cls):
        return SolverInterface.Get_required_arguments()

    @classmethod
    def Get_optional_arguments(cls):
        optional_arguments = SolverInterface.Get_optional_arguments()
        optional_arguments['momentum'] = 0.0
        return optional_arguments


# register the locally specified solver
SolverInterface.Register_solver(SGDSolver)
SolverInterface.Register_solver(AdagradSolver)
SolverInterface.Register_solver(NesterovSolver)


Get_solver_class = SolverInterface.Get_solver_class
Get_caffe_solver_class = SolverInterface.Get_caffe_solver_class
