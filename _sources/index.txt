.. barrista documentation master file, created by
   sphinx-quickstart on Tue Jun 23 17:47:50 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to barrista's documentation!
====================================

Barrista will serve you your `caffe <https://github.com/BVLC/caffe>`_
right! It is a Python library that offers comparable access
to `caffe` as `keras <http://keras.io>`_ or
`Theano <http://deeplearning.net/software/theano/>`_.

Design your nets with the full power of `caffe` (and guaranteed correctness and
completeness) while not using any `.prototxt` files and not messing with
configurations; and use the flexibility and convenience of Python
by using Callbacks for properly controlling and monitoring
your training procedure!
We achieve this by inspecting the protobuf objects that are used to configure
caffe, and exposing their functionality (and some convenient additions)
nicely. And the best: there is full compatibility to import and export
`.prototxt` files so you can start working with your specified DCNNs right
away or get some from the
`model zoo <http://caffe.berkeleyvision.org/model_zoo.html>`_!
It runs on every platform caffe runs on, and can be used with Python 2 and 3.

.. figure:: _static/barrista.jpg
   :scale: 50 %
   :alt: map to buried treasure
   :align: right

   `Photo by Liz Clayton <https://commons.wikimedia.org/wiki/File:Gwilym_Davies.jpg>`_,
   `CC <http://creativecommons.org/licenses/by/3.0/>`_.


Contents:

.. toctree::
   :maxdepth: 4

   setup
   usage
   barrista
   about


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
