# -*- coding: utf-8 -*-
# pylint: disable=F0401, E0611, C0103, R0902, R0913, R0914, E1101
"""
This module contains classes and functions to design networks with caffe.

For the layer constructing functions, ``None`` is used to mark a parameter
as unset (this is important in protobuf)!
"""

import copy as _copy
import inspect as _inspect
import logging as _logging
import subprocess as _subprocess
import itertools as _itertools
from tempfile import NamedTemporaryFile as _NamedTemporaryFile
import os as _os
from operator import itemgetter as _itemgetter
import cv2 as _cv2
import google.protobuf.text_format as _gprototext

# CAREFUL! This must be imported before any caffe-related import!
from .initialization import init as _init
from .config import LAYER_TYPES as _LAYER_TYPES
from .config import CAFFE_BIN_FOLDER as _CAFFE_BIN_FOLDER
from .tools import chunks as _chunks, pbufToPyEnum as _pbufToPyEnum

import caffe.proto.caffe_pb2 as _caffe_pb2
try:
    import caffe.draw as _draw
except ImportError:
    _draw = None
from .net import Net as _Net


#: Expose all detailed properties from the caffe prototxt for
#: in-Python usage.
PROTODETAIL = _caffe_pb2

#: Whether this caffe version supports the `propagate_down` layer property.
_HAS_PROPAGATE_DOWN = hasattr(_caffe_pb2.LayerParameter,
                              'propagate_down')
#: Whether this caffe version supports 'BlobShape'.
_HAS_BLOB_SHAPE = hasattr(_caffe_pb2, 'BlobShape')

_init()
_LOGGER = _logging.getLogger(__name__)
if _draw is None:
    _LOGGER.warn('Could not import caffe.draw! Drawing is not available. ' +
                 'Probably this is due to the package pydot not being ' +
                 'available.')


#: Describes the phase a network is in (TRAIN or TEST). It may only be set
#: when instantiating the network. If phase is TEST, some memory optimizations
#: are done, that prohibit training the network. The other way around, it is
#: no problem to test a network in TRAIN mode, but it will be less efficient.
Phase = _pbufToPyEnum(_caffe_pb2.Phase)


class NetSpecification(object):

    r"""
    Represents a network specification.

    Initializes the fields with similar semantics as the
    prototxt files. An important difference is, that if ``predict_inputs`` and
    ``predict_input_shapes`` are specified, this corresponds to two virtual
    network specifications. One, where the stage is set as specified by the
    ``stages`` parameter, and one
    where the stage is set to 'predict'. Both networks share the same weights.
    When instantiated, the 'predict' network will automatically be used by
    the :py:func:`barrista.net.Net.predict` method.

    .. graph:: network_stages_and_phases

       subgraph cluster_fit {
           phase_train_fit [label="phase: train"];
           phase_test_fit [label="phase: test"];
           label="stage: fit";
           graph[style=dotted];
       }
       subgraph cluster_predict {
           phase_test_predict [label="phase: test"];
           label="stage: predict";
           graph[style=dotted];
       }

    To get the plain specification of the network in 'predict' mode, use the
    method
    :py:func:`barrista.design.NetSpecification.get_predict_net_specification`.

    :param input_shape: list(list(int)).
      The input shape specification of
      the network. The length of each of the sub-lists must be > 0.

    :param inputs: list(string) or None.
      The names of the network inputs. The length must match the length
      of the ``input_shape`` list. If ``None`` is used, this is initialized
      as ``['data']``.

    :param layers: list(:py:class:`barrista.design.LayerSpecification`) or None.
      The layer specifications. If None is used, this is initialized as ``[]``.

    :param force_backward: bool.
      Whether to force a backward pass for all layers only useful during
      training.

    :param phase: :py:data:`barrista.design.Phase`.
      The phase to have the network in.

    :param level: int.
      The level of the network. Can be used to in- or exclude layers depending
      on the level.

    :param stages: list(string) or None.
      The stages the network is in. Can be used to in- or exclude layers
      depending on the stages. By default, 'fit' and 'predict' stages are
      used, as mentioned before. If ``None`` is specified, this is initialized
      as ``['fit']``.

    :param debug_info: bool.
      If set to True, give additional debug output on the console.

    :param name: string.
      The name of the network.

    :param predict_inputs: list(string) or None.
      If set, will be used as ``inputs`` for a network with the same
      specification, except ``stages=['predict']``. This will then
      automatically used after instantiation for predicting inputs.

    :param predict_input_shapes: list(list(int)) or None.
      If set, will be used together with ``predict_inputs`` as stated above.
    """

    def __init__(self,
                 input_shape,
                 inputs=None,
                 layers=None,
                 force_backward=False,
                 phase=Phase.TEST,
                 level=0,
                 stages=None,
                 debug_info=False,
                 name='DCNN',
                 predict_inputs=None,
                 predict_input_shapes=None):
        """See class documentation."""
        if inputs is None:
            inputs = ['data']
        if layers is None:
            layers = []
        if stages is None:
            stages = ['fit']
        assert len(inputs) == len(input_shape)
        for shapelist in input_shape:
            assert len(shapelist) > 0
        if predict_inputs is not None or predict_input_shapes is not None:
            assert predict_inputs is not None
            assert predict_input_shapes is not None
            assert len(predict_inputs) == len(predict_input_shapes)
            for shape in predict_input_shapes:
                assert len(shape) > 0 and len(shape) < 5
        self.input_shape = input_shape
        self.inputs = inputs
        self.layers = layers
        self.force_backward = force_backward
        self.phase = phase
        self.level = level
        self.stages = stages
        self.debug_info = debug_info
        self.name = name
        self.predict_inputs = predict_inputs
        self.predict_input_shapes = predict_input_shapes

    def get_predict_net_specification(self):
        """Get the plain network specification with ``stages=['predict']``."""
        assert (self.predict_inputs is not None and
                self.predict_input_shapes is not None)
        return NetSpecification(self.predict_input_shapes,
                                self.predict_inputs,
                                self.layers,
                                False,
                                Phase.TEST,
                                self.level,
                                ['predict'],
                                self.debug_info,
                                self.name)

    def to_pbuf_message(self):
        r"""
        Create a plain protobuf message from this object.

        Since this object is not derived from the protobuf objects, it is
        necessary to have this converter. Deriving is discouraged by the
        protobuf documentation and this object offers a lot more functionality.
        """
        pblayers = []
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                pblayers.append(layer.to_pbuf_message(idx, None, self.inputs))
            else:
                pblayers.append(layer.to_pbuf_message(idx,
                                                      self.layers[idx-1],
                                                      self.inputs))
        pbstate = _caffe_pb2.NetState(phase=self.phase,
                                      level=self.level,
                                      stage=self.stages)
        if _HAS_BLOB_SHAPE:
            pbinput_shape = [_caffe_pb2.BlobShape(dim=dims)
                             for dims in self.input_shape]
            netmessage = _caffe_pb2.NetParameter(name=self.name,
                                                 input=self.inputs,
                                                 input_shape=pbinput_shape,
                                                 layer=pblayers,
                                                 force_backward=self.force_backward,  # noqa
                                                 state=pbstate,
                                                 debug_info=self.debug_info)
        else:
            shapes_to_use = []
            for shape in self.input_shape:
                if len(shape) != 4:
                    shape += [1] * (4 - len(shape))
                shapes_to_use.append(shape)
            netmessage = _caffe_pb2.NetParameter(name=self.name,
                                                 input=self.inputs,
                                                 input_dim=_itertools.chain(*self.input_shape),
                                                 layer=pblayers,
                                                 force_backward=self.force_backward,  # noqa
                                                 state=pbstate,
                                                 debug_info=self.debug_info)
        assert netmessage.IsInitialized()
        return netmessage

    def to_prototxt(self, output_filename=None):
        r"""
        Create a plain, human readable, prototxt representation.

        If ``output_filename`` is set, the resulting text is written into
        that file, as well as returned. Otherwise, only the text is returned.

        Layers will be automatically wired together, if their ``bottoms`` or
        ``tops`` are not set. If the have multiple inputs or outputs, you
        will have to take care of that (there is no way of inferring
        the semantics then).
        """
        messagestr = _gprototext.MessageToString(self.to_pbuf_message())
        if output_filename is not None:
            with open(output_filename, 'w') as outf:
                outf.write(messagestr)
        return messagestr

    @staticmethod
    def from_prototxt(text=None, filename=None):
        r"""
        Create an :py:class:`NetSpecification` object from a text spec.

        Either ``text`` or ``filename`` may be set, and is accordingly used.
        Files may be of any caffe prototxt version.
        """
        # Check if the user erroneously specified a filename as text.
        if text is not None:
            if _os.linesep not in text:
                if _os.path.exists(text):
                    _LOGGER.warn('You probably mistakenly specified a filename '
                                 'as text: "%s"! Trying to recover...', text)
                    filename = text
                    text = None
        if filename is not None:
            assert text is None
            # Do a conversion if necessary.
            with _NamedTemporaryFile(mode='r', suffix='.prototxt') as tmpfile:
                net_upgrader_exec = _os.path.join(_CAFFE_BIN_FOLDER,
                                                  'upgrade_net_proto_text')
                assert _os.path.exists(net_upgrader_exec),\
                    ("The executable 'upgrade_net_proto_text' was not found "
                     "in your _CAFFE_BIN_FOLDER! Please set it from the "
                     "module `barrista.config`. The current folder is set "
                     "to: " + _CAFFE_BIN_FOLDER + ".")
                _subprocess.check_call([net_upgrader_exec,
                                        filename,
                                        tmpfile.name])
                text = tmpfile.read()
        message = _caffe_pb2.NetParameter()
        _gprototext.Merge(text, message)
        # Check for completeness of the parsing process.
        fields = message.ListFields()
        for fielddesc in map(_itemgetter(0), fields):  # pylint: disable=W0141
            if fielddesc.name not in ['name',
                                      'input_shape',
                                      'debug_info',
                                      'input',
                                      'input_dim',
                                      'layer',
                                      'force_backward',
                                      'state']:
                _LOGGER.warn('Parsed net prototxt contained unknown field ' +
                             fielddesc.name + '. Ignored.')
        if len(message.input_dim) > 0:
            _LOGGER.warn('The loaded prototxt contains `input_dim` fields. '
                         'They are deprecated! Use `input_shape` instead.')
            if _HAS_BLOB_SHAPE:
                assert len(message.input_shape) == 0
            assert len(message.input_dim) % 4 == 0
            input_shape = _copy.deepcopy(list(_chunks(message.input_dim, 4)))
        else:
            input_shape = _copy.deepcopy([bshape.dim for
                                          bshape in message.input_shape])
        inputs = _copy.deepcopy(message.input)
        layerspecs = [LayerSpecification.from_pbuf_message(layer)
                      for layer in message.layer]
        pbforcebw = message.force_backward
        phase = message.state.phase
        level = message.state.level
        stages = _copy.deepcopy(message.state.stage)
        debug_info = message.debug_info
        name = message.name
        spec = NetSpecification(input_shape,
                                inputs,
                                layerspecs,
                                pbforcebw,
                                phase,
                                level,
                                stages,
                                debug_info,
                                name)
        return spec

    def copy(self):
        r"""
        Create a deep copy of this object.

        All layers are deep copied, so modifying layer objects of the old
        object does not influence the copied one!
        """
        return _copy.deepcopy(self)

    def instantiate(self):
        """Create an instantiated net with the current object configuration."""
        # Write spec to a temporary file.
        with _NamedTemporaryFile(mode='w', suffix='.prototxt') as tmpfile:
            tmpfile.write(self.to_prototxt())
            tmpfile.flush()
            _specification = self.copy()
            net = _Net(tmpfile.name,
                       specification=_specification)
        return net

    def visualize(self,
                  layout_dir='LR',
                  display=False):
        """
        Create and optionally display an image of the net structure.

        :param layout_dir: string in ['LR', 'TB', 'BT'].
          Short string for graph layout direction.

        :param display: bool.
          If set to ``True``, displays the graphic in a window. Press enter
          to close it.

        :returns: 3D numpy array.
          Graphic of the visualization as (H, W, C) image in BGR format.
        """
        if _draw is None:
            raise Exception('Drawing is not available!')
        with _NamedTemporaryFile(mode='w+b', suffix='.png') as tmpfile:
            _draw.draw_net_to_file(self.to_pbuf_message(),
                                   tmpfile.name,
                                   rankdir=layout_dir)
            result_image = _cv2.imread(tmpfile.name)
            assert result_image is not None
        if display:
            _cv2.imshow(self.name, result_image)
            _cv2.waitKey(0)
            _cv2.destroyWindow(self.name)
        return result_image


class LayerSpecification(object):

    r"""
    Describes one caffe layer.

    :param bottoms: list(string) or None.
      If set, specifies the inputs. If unset, will be automatically wired
      to the preceeding layer ``top[0]``.

    :param tops: list(string) or None.
      If set, specifies the top names of this layer. If unset, will be set to
      the layer name.

    :param name: string or None.
      If set, gives the name of the layer. Otherwise, use ``_layer_{idx}``,
      where ``idx`` specifies the layer index.

    :param phase: int or None.
      If set, specifies the layer phase.

    :param include_phase: :py:data:`barrista.design.Phase` or None.
      Only include this layer in the given phase.

    :param include_stages: string or None.
      Only include this layer if *all* stages are present.

    :param include_min_level: int or None.
      Only include this layer, if the network level is >= this value.

    :param include_max_level: int or None.
      Only include this layer, if the network level is <= this value.

    :param params: list(:py:data:`barrista.design.ParamSpec`) or None.
      Multipliers for learning rate and weight decay for the layer parameters.

    :param propagate_down: list(bool) or None.
      Specifies on which bottoms the backpropagation should be skipped.
      Must be either 0 or equal to the number of bottoms. If ``None`` is
      specified, this is initialized as ``[]``. Not available in all
      caffe versions!

    :param loss_param: :py:data:`barrista.design.LossParameter` or None.
      Specifies optional ignore labels and normalization for the loss.

    :param loss_weights: list(float) or None.
      The amount of weight to assign each top blob in the objective. If ``None``
      is specified, this is initialized as ``[]``.
    """

    def __init__(self,
                 bottoms=None,
                 tops=None,
                 name=None,
                 phase=None,
                 include_phase=None,
                 include_stages=None,
                 include_min_level=None,
                 include_max_level=None,
                 params=None,
                 propagate_down=None,
                 loss_param=None,
                 loss_weights=None):
        """See class documentation."""
        self.bottoms = bottoms
        self.tops = tops
        self.name = name
        self.phase = phase
        self.include_phase = include_phase
        self.include_stages = include_stages
        self.include_min_level = include_min_level
        self.include_max_level = include_max_level
        self.params = params
        if propagate_down is None:
            propagate_down = []
        else:
            if not _HAS_PROPAGATE_DOWN:
                raise Exception("This caffe version does not support the "
                                "`propagate_down` layer property!")
        self.propagate_down = propagate_down
        self.loss_param = loss_param
        if loss_weights is None:
            loss_weights = []
        self.loss_weights = loss_weights
        self.type = None
        self._additional_parameters = []

    def __eq__(self, other):
        """Deep equality comparison for all properties."""
        for attrname in ['bottoms',
                         'tops',
                         'name',
                         'phase',
                         'include_phase',
                         'include_stages',
                         'include_min_level',
                         'include_max_level',
                         'params',
                         'propagate_down',
                         'loss_param',
                         'loss_weights',
                         'type',
                         '_additional_parameters']:
            if not getattr(self, attrname) == getattr(other, attrname):
                return False
        return True

    @staticmethod
    def from_pbuf_message(message):
        r"""Create a LayerSpecification object from a protobuf message."""
        bottoms = _copy.deepcopy(message.bottom)
        tops = _copy.deepcopy(message.top)
        name = _copy.deepcopy(message.name)
        include_phase = None
        include_stages = None
        include_min_level = None
        include_max_level = None
        if len(message.include) > 0:
            msg0fields = message.include[0].ListFields()
            fieldnames = [fld[0].name for fld in msg0fields]
            include_phase = None
            if 'phase' in fieldnames:
                include_phase = _copy.deepcopy(message.include[0].phase)
            if 'stage' in fieldnames:
                include_stages = _copy.deepcopy(message.include[0].stage)
            if 'min_level' in fieldnames:
                include_min_level = _copy.deepcopy(message.include[0].min_level)
            if 'max_level' in fieldnames:
                include_max_level = _copy.deepcopy(message.include[0].max_level)
            if len(message.include) > 1:
                _LOGGER.warn('Layer %s has include specifications that can '
                             'not be modeled in this tool! Ignoring.', name)
        fields = message.ListFields()
        fieldnames = [fld[0].name for fld in fields]
        phase = None if 'phase' not in fieldnames else _copy.deepcopy(
            message.phase)
        params = None if 'param' not in fieldnames else _copy.deepcopy(
            message.param)
        if _HAS_PROPAGATE_DOWN:
            propagate_down = _copy.deepcopy(message.propagate_down)
        else:
            propagate_down = None
        loss_param = (None if 'loss_param' not in fieldnames else
                      _copy.deepcopy(message.loss_param))
        loss_weights = _copy.deepcopy(message.loss_weight)
        spec = LayerSpecification(bottoms,
                                  tops,
                                  name,
                                  phase,
                                  include_phase,
                                  include_stages,
                                  include_min_level,
                                  include_max_level,
                                  params,
                                  propagate_down,
                                  loss_param,
                                  loss_weights)
        # Get type and type-dependent parameters.
        spec.type = message.type
        fields = message.ListFields()
        for fielddesc, fieldval in fields:
            if fielddesc.name not in ['name',
                                      'bottom',
                                      'type',
                                      'top',
                                      'phase',
                                      'include',
                                      'param',
                                      'propagate_down',
                                      'loss_param',
                                      'loss_weight']:
                setattr(spec, fielddesc.name, _copy.deepcopy(fieldval))
                # pylint: disable=W0212
                spec._additional_parameters.append(fielddesc.name)
        return spec

    def to_pbuf_message(self,  # pylint: disable=R0912, R0915
                        layerindex,
                        preceeding_layer,
                        net_input):
        r"""
        Create a protobuf specification of this layer.

        It automatically wires together preceeding and following layers,
        if ``tops`` or ``bottoms`` are not set. This does not work with
        multiple in- or outputs.

        :param layerindex: int >= 0.
          The index of this layer. Is used to generate the layer name.

        :param preceeding_layer: :class:`barrista.design.LayerSpecification`.
          The preceeding layer to create the wiring with.

        :param net_input: string.
          The name of the network input (used for the first layer input).
        """
        assert layerindex >= 0
        assert self.type is not None
        if layerindex != 0:
            assert preceeding_layer is not None
        # Prepare the arguments.
        kwargs = dict()
        # Bottom.
        pbbottom = []
        omit_bottom = False
        if self.bottoms is None:
            if preceeding_layer is None:
                if net_input is not None and len(net_input) > 0:
                    pbbottom = [net_input[0]]
                else:
                    omit_bottom = True
            else:
                if preceeding_layer.tops is None:
                    if preceeding_layer.name is None:
                        pbbottom = ['_layer_{0}'.format(layerindex - 1)]
                    else:
                        pbbottom = [preceeding_layer.name]
                else:
                    pbbottom = [preceeding_layer.tops[0]]
        else:
            preplist = self.bottoms[:]
            mapidx = 0
            for btidx, btname in enumerate(preplist):
                if btname is None:
                    if preceeding_layer.tops is not None:
                        preplist[btidx] = preceeding_layer.tops[mapidx]
                    else:
                        preplist[btidx] = '_layer_{0}'.format(layerindex - 1)
                    mapidx += 1
            pbbottom = preplist
        if not omit_bottom:
            kwargs['bottom'] = pbbottom
        # Top.
        pbtop = []
        if self.tops is None:
            if self.name is None:
                pbtop = ['_layer_{0}'.format(layerindex)]
            else:
                pbtop = [self.name]
        else:
            pbtop = self.tops
        kwargs['top'] = pbtop
        # Name.
        pbname = self.name
        if pbname is None:
            pbname = '_layer_{0}'.format(layerindex)
        kwargs['name'] = pbname
        if self.phase is not None:
            kwargs['phase'] = self.phase
        # include.
        include_kwargs = dict()
        if self.include_phase is not None:
            include_kwargs['phase'] = self.include_phase
        if self.include_stages is not None and len(self.include_stages) > 0:
            include_kwargs['stage'] = self.include_stages
        if self.include_min_level is not None:
            include_kwargs['min_level'] = self.include_min_level
        if self.include_max_level is not None:
            include_kwargs['max_level'] = self.include_max_level
        if len(include_kwargs) > 0:
            kwargs['include'] = [_caffe_pb2.NetStateRule(**include_kwargs)]
        kwargs['type'] = self.type
        if self.params is not None:
            kwargs['param'] = self.params
        if _HAS_PROPAGATE_DOWN:
            kwargs['propagate_down'] = self.propagate_down
        if self.loss_param is not None:
            kwargs['loss_param'] = self.loss_param
        kwargs['loss_weight'] = self.loss_weights
        for add_pname in self._additional_parameters:
            kwargs[add_pname] = getattr(self, add_pname)
        layerMessage = _caffe_pb2.LayerParameter(**kwargs)
        assert layerMessage.IsInitialized()
        return layerMessage


# Generate the layers.
for _layerkey in list(_LAYER_TYPES.keys()):
    # Construct the layer constructor.
    _parameters = []
    # Get the parameters for the standard layer function.
    _layerArgSpec = _inspect.getargspec(LayerSpecification.__init__)
    _without_defaults = len(_layerArgSpec.args) - len(_layerArgSpec.defaults)
    for _idx in range(1, len(_layerArgSpec.args)):
        _parameters.append((_layerArgSpec.args[_idx],
                            _idx >= _without_defaults,
                            'LayerSpecification',
                            _layerArgSpec.args[_idx]))
    # Get the layer specific parameters by inspecting the protobuf objects.
    _layer_error = False
    for _param_obj in _LAYER_TYPES[_layerkey]:
        assert _param_obj.endswith('Parameter'),\
            ('Only add the ...Parameter '
             'objects as layer parameters. Their name must end with '
             '"Parameter". All other parameters concerning the layer '
             'must be explicitely coded.')
        _obj_type = None
        try:
            exec('_obj_type = _caffe_pb2.' + _param_obj)  # pylint: disable=W0122
        except AttributeError as nfe:
            print(("[WARNING] Parameter {} not found in caffe proto "
                   "configuration of {}! Adjust your barrista.config! "
                   "Skipping layer!").format(_param_obj, _layerkey))
            _layer_error = True
            break
        # This is guaranteed to work, because the additional parameter objects
        # for caffe MUST only have optional parameters.
        exec('_obj_instance = _obj_type()')  # pylint: disable=W0122
        for _fieldname in list(_obj_type.__dict__.keys()):
            if isinstance(_obj_type.__dict__[_fieldname], property):
                _parameters.append((_param_obj[:-9] + '_' + _fieldname,
                                    True,
                                    _param_obj,
                                    _fieldname))
    if _layer_error:
        continue
    # Analyzed the layers. Generating layer objects...
    _layer_prefix = _layerkey
    _func_spec = 'def {_layer_prefix}Layer('.format(**locals())
    for _ptpl in _parameters:
        _func_spec += _ptpl[0]
        if _ptpl[1]:
            _func_spec += '=None'  # '={0}().{1}'.format(_ptpl[2], _ptpl[3])
        _func_spec += ','
    _func_spec = _func_spec[:-1]
    _func_spec += '):' + _os.linesep
    _func_spec += '    _ret_obj = LayerSpecification({0})'.format(
        ', '.join(_layerArgSpec.args[1:])) + _os.linesep
    for _param_obj in _LAYER_TYPES[_layerkey]:
        _obj_type = None
        exec('_obj_type = _caffe_pb2.' + _param_obj)  # pylint: disable=W0122
        # identify the property name for this parameter.
        _detected = False
        for _propname in list(_caffe_pb2.LayerParameter.__dict__.keys()):
            if isinstance(getattr(_caffe_pb2.LayerParameter(), _propname),
                          _obj_type):
                _detected = True
                break
        assert _detected, ('The parameter name of the layer property {0} ' +
                           'could not be found!').format(_param_obj)
        _func_spec += '    _ret_obj._additional_parameters.append("{_propname}")'.format(**locals()) + _os.linesep  # noqa
        _func_spec += '    {_propname}_dummy = _caffe_pb2.{_param_obj}()'.format(**locals()) + _os.linesep  # noqa
        _func_spec += '    {_propname}_kwargs = dict()'.format(**locals()) + _os.linesep  # noqa
        for _ptpl in _parameters:
            if _ptpl[2] == _param_obj:
                # Stay compatible with older caffe version where some fields
                # were scalars that are now repeated fields.
                _func_spec += (r'''    if {_ptpl[0]} is not None:
        try:
            _ = {_ptpl[0]} + 1  # Python version independent integer check
            if (hasattr(getattr({_propname}_dummy, '{_ptpl[3]}'), 'append')):
                # The attribute is list typed now, so convert.
                {_ptpl[0]} = [{_ptpl[0]}]
        except:
            # Nothing to do here.
            pass
        {_propname}_kwargs["{_ptpl[3]}"] = {_ptpl[0]}{_os.linesep}''')\
            .format(**locals())
        _func_spec += (
            '    _ret_obj.{_propname} = _caffe_pb2.{_param_obj}(**{_propname}_kwargs){_os.linesep}'
            ).format(**locals())  # noqa
    _func_spec += '    _ret_obj.type = "{}"'.format(_layerkey) + _os.linesep
    _func_spec += '    return _ret_obj'
    _LOGGER.debug(_func_spec)
    exec(_func_spec)  # pylint: disable=W0122
