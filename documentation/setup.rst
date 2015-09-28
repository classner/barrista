Setup
=====

============
Requirements
============

This package has the following requirements:

* numpy,
* caffe, rc2 or newer,  built with the python interface,
* OpenCV (for automatic rescaling),
* setuptools (for installation and building the documentation),
* sphinx (for building the documentation).

==================
Caffe modification
==================

While barrista nearly only 'serves' unmodified caffe right,
some beans must be added for perfect flavor. To achieve this,
open your `caffe/python/caffe/_caffe.cpp` file and add
the following lines::

    // In the upper part, add this function (place doesn't matter
    // much)
    void Net_load_blobs_from(Net<Dtype>& net, string filename) {
      CheckFile(filename);
      net.CopyTrainedLayersFrom(filename);
    }
    ...
    // In the lower part, change the declaration of the
    // caffe net by adding:
    bp::class_<Net<Dtype>, shared_ptr<Net<Dtype> >, boost::noncopyable >("Net",
      bp::no_init)
      .def("__init__", bp::make_constructor(&Net_Init))
      .def("__init__", bp::make_constructor(&Net_Init_Load))
      .def("_forward", &Net<Dtype>::ForwardFromTo)
      .def("_backward", &Net<Dtype>::BackwardFromTo)
      .def("reshape", &Net<Dtype>::Reshape)
      .def("load_blobs_from", &Net_load_blobs_from) // <-- this line!
      // The cast is to select a particular overload.
      ...

=============
Configuration
=============

Make sure to adjust `config.py` to your needs. The
variables `CAFFE_PYTHON_FOLDER` and `CAFFE_BIN_FOLDER`
must be changed to their appropriate values. You can
either do this before installation and never have to
care for it again, or you can 'hot-configure' the module
during use, by running::

    import barrista.config
    # This must be done before importing any other submodule.
    barrista.config.CAFFE_PYTHON_FOLDER = 'your/path'
    barrista.config.CAFFE_BIN_FOLDER = 'your/bin/path'
    import barrista.design
    ...

Alternatively, you can set both variables as environment variables.
If they are present, the variables will be initialized accordingly.

=======
Testing
=======

You can run the tests to verify proper configuration and compatibility by
executing::

    python setup.py test

============
Installation
============

You can install the project to your python installation by running::

    python setup.py install

If you want to build the documentation, run::

    python setup.py build_sphinx

Should this command fail with an error message along the lines of
'`build_sphinx` is an invalid command', just upgrade your distutils
installation by running `pip install --upgrade distutils` and it
should work.


=============================
Registering additional layers
=============================

The layer types and their parameters can, unfortunately, not be inferred
from the caffe protobuf protocols. Your standard barrista knows all the
popularly used layer types that come with the defaut caffe installation, but
if you extend caffe, it is necessary to register your new layers
by hand. There are again two possibilities:

#. edit the file `config.py` by hand,
#. register the layer types on-the-fly during use.

The responsible object is `barrista.config.LAYER_TYPES`. It is a dictionary
with the layer name strings as keys, and a list of names of protobuf objects
that are the layer's parameters, e.g.,::

    'Convolution': ['ConvolutionParameter']

is responsibe for registering the convolution layer.
You can simply edit `config.py` and add your own layers, or can add them on the
fly as follows::

    import barrista.config
    # This must be done before importing any other submodule!
    barrista.config.LAYER_TYPES['Convolution'] = ['ConvolutionParameter']
    import barrista.design
    ...
