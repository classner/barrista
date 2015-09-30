Setup
=====

============
Requirements
============

This package has the following requirements:

* caffe, rc2 or newer,  built with the Python interface,
* OpenCV (for automatic rescaling),
* setuptools (for installation and building the documentation),
* sphinx (for building the documentation).

===================
Caffe modifications
===================

While barrista only 'serves' caffe right, some beans must be added for the
perfect flavor. This translates to just a few lines of C++ code that must
be changed in the core library.

We offer a .patch file for the caffe rc2 and the version with added
parallelization support and the new solvers from September 2015. But even
if you are not using one of these version, don't worry, you will be able to
easily add them by hand.

The patch files are located in the barrista folder `patches` and can be applied
(for example) by getting `pypatch <https://pypi.python.org/pypi/pypatch/0.5.1>`_
and executing::

    pypatch apply ../path/to/barrista/patches/patch_[X].patch path/to/caffe/root

If there is no patch available for your very caffe version, you should still
be able to quickly find out what lines to change by having a look at the
`.patch` files.

=============
Configuration
=============

Now that your `caffe` is ready, we will setup `barrista` so that it knows
what caffe to work with!

There are three ways to do this:

#. edit the source files,
#. set environment variables,
#. do it on-the-fly within your code.

The first possibility is not as `clean` as the others (since, e.g., a
barrista update might break your config), but in case you want to do it:
the file ``barrista/config.py`` is all yours! It contains all relevant
information (the meaning of the variable ``LAYER_TYPES`` is discussed in
detail in the section :ref:`registering-layers`).

A clean way to let barrista know about the location of the `caffe` to use is,
to use the two environment variables :py:data:`CAFFE_PYTHON_FOLDER` and
:py:data:`CAFFE_BIN_FOLDER`. Whereas the first one points to the folder where
caffe's Python module is located (usually ``caffe/python``), the second one points
to the folder where all caffe executables are stored after having run
the installation (usually ``caffe/build/install/bin``). This folder must contain
the executable ``upgrade_net_proto_text``.

If you are working with many different `caffe` builds or want to swap them
on the fly, you can also do this easily! In this case::

    # Import the barrista's config module before any of the other modules!
    import barrista.config
    # Change the two variable values to your liking:
    barrista.config.CAFFE_PYTHON_FOLDER = 'a/crazy/folder'
    barrista.config.CAFFE_BIN_FOLDER = 'another/funny/one'
    # Now use your configured barrista:
    import barrista.design
    # ...

=======
Testing
=======

Before installing barrista, you can (and should) run the tests to verify
proper configuration and compatibility by executing::

    python setup.py test

============
Installation
============

You can install the project to your python installation by running::

    python setup.py install

or, to be able to modify the software in it's current folder while using it,
run::

    python setup.py develop

If you want to build the documentation, run::

    python setup.py build_sphinx

Should this command fail with an error message along the lines of
'`build_sphinx` is an invalid command', just upgrade your distutils
installation by running ``pip install --upgrade distutils`` and it
should work.

.. _registering-layers:

=============================
Registering additional layers
=============================

The layer types and their parameters can, unfortunately, not be inferred
from the caffe protobuf protocols in a fully automatic manner.
Your standard barrista knows all the popularly used layer types that come with
the default caffe installation, but if you extend caffe, it is necessary
to register your new layers by hand. There are again two possibilities:

#. edit the file ``config.py`` by hand,
#. register the layer types on-the-fly during use.

The responsible object is :py:data:`barrista.config.LAYER_TYPES`. It is a dictionary
with the layer name strings as keys, and a list of names of protobuf objects
that are the layer's parameters, e.g.,::

    'Convolution': ['ConvolutionParameter']

is responsibe for registering the convolution layer.
You can simply edit ``config.py`` and add your own layers, or add them on the
fly as follows::

    import barrista.config
    # This must be done before importing any other submodule!
    barrista.config.LAYER_TYPES['Convolution'] = ['ConvolutionParameter']
    import barrista.design
    ...
