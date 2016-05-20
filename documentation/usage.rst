Using `barrista`
================

This file gives a quite comprehensive walkthrough through nearly all
features offered by barrista. If you want to get your hands dirty right away,
there is a comprehensive example of a VGG-like net being trained and applied
in the file ``example.py`` in the root folder of the barrista package.

==================================
Importing and configuring barrista
==================================

If you have `caffe` on your path, you can use barrista right away and
include and use any of its submodules. Otherwise, you can configure it
to use a specific `caffe` version on the fly as follows::

    import barrista.config
    # This must be done before importing any other submodule.
    barrista.config.CAFFE_PYTHON_FOLDER = 'your/path'
    barrista.config.CAFFE_BIN_FOLDER = 'your/bin/path'
    import barrista.design
    ...

For an exact description of the two parameters, see
:py:data:`barrista.config.CAFFE_PYTHON_FOLDER` and
:py:data:`barrista.config.CAFFE_BIN_FOLDER`.

================================
Creating a network specification
================================

The module :py:mod:`barrista.design` contains methods and classes to
design `caffe` models. We will use it in the following example to create
a simple, `VGG`-like model::

    import barrista.design as design
    from barrista.design import (ConvolutionLayer, ReLULayer, PoolingLayer,
                                  DropoutLayer, InnerProductLayer,
                                  SoftmaxLayer, SoftmaxWithLossLayer,
                                  AccuracyLayer)

    # The only required parameter is a list of lists with the input shape
    # specification for the network. In this case, we also specify names
    # for the inputs layers.
    netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                      inputs=['data', 'annotations'])

    layers = []
    conv_params = {'Convolution_kernel_size': 3,
                   'Convolution_num_output': 32,
                   'Convolution_pad': 1}

    # If not specified, the first top blob for each layer is automatically
    # wired with the first bottom of the preceeding layer. If your are using
    # multi-in/out layers, you have to manually specify tops and bottoms.

    layers.append(ConvolutionLayer(**conv_params))
    layers.append(ReLULayer())
    layers.append(ConvolutionLayer(**conv_params))
    layers.append(ReLULayer())
    layers.append(PoolingLayer(Pooling_kernel_size=2))
    layers.append(DropoutLayer(Dropout_dropout_ratio=0.25))

    conv_params['Convolution_num_output'] = 64
    layers.append(ConvolutionLayer(**conv_params))
    layers.append(ReLULayer())
    layers.append(ConvolutionLayer(**conv_params))
    layers.append(ReLULayer())
    layers.append(PoolingLayer(Pooling_kernel_size=2))
    layers.append(DropoutLayer(Dropout_dropout_ratio=0.25))

    layers.append(InnerProductLayer(InnerProduct_num_output=256))
    layers.append(ReLULayer())
    layers.append(DropoutLayer(Dropout_dropout_ratio=0.25))

    layers.append(InnerProductLayer(InnerProduct_num_output=10))
    layers.append(SoftmaxLayer())

    netspec.layers.extend(layers)

The layer names are exactly the same as in the prototxt format. All direct
parameters for a layer can be set by using it's constructor or later be set
as it's object property. If you have to use sub-objects (or rather messages,
in prototxt-speak), they are all available from the object
:py:data:`barrista.design.PROTODETAIL`.

You can now inspect the specification and programatically change its parameters.
To get the prototxt representation, use the method
:py:func:`barrista.design.NetSpecification.to_prototxt`::

    print(netspec.to_prototxt())

The method has an additional parameter ``output_filename`` that can be used to
directly create prototxt files::

    netspec.to_prototxt(output_filename='test.prototxt')

=====================
Visualizing a network
=====================

It is possible to visualize a network specification or an instantiated
network by calling its :py:func:`barrista.design.NetSpecification.visualize`
or :py:func:`barrista.net.Net.visualize` function. It is possible to directly
display it or write it to a file::

    # Create the visualization and display it.
    viz = netspec.visualize(display=True)
    # Write it to a file.
    import cv2
    cv2.imwrite('/tmp/test.png', viz)

=================================
Importing a network specification
=================================

You can work with all your already prepared prototxt files as well! Use the
method :py:func:`barrista.design.NetSpecification.from_prototxt` to load
any valid caffe model (of any version!) and inspect and modify it in this
framework::

    netspec_reloaded = design.NetSpecification.from_prototxt(filename='test.prototxt')

===============
Using a network
===============

However, apart from diagnostic or logging
purposes, it is not necessary to work with prototxt specifications any more.
Simply run::

    net = netspec.instantiate()

to get a fully working network object. It is subclassed from the
``caffe.Net``, so it comes with all the methods you are familiar with. But
be prepared for some more convenience! You can set cpu or gpu mode by
using :py:func:`barrista.net.set_mode_cpu` and
:py:func:`barrista.net.set_mode_gpu`.

Loading parameters
~~~~~~~~~~~~~~~~~~

With this, the blobs can be loaded as::

  net.load_blobs_from('your/path/to/blobs.caffemodel')

and to restore a solver, use::

  solver.restore('your/path/to/xyz.solverstate', net)

**CAUTION**: The blobs are stored in the ``.caffemodel``s by name. Blobs will be
matched to network layers with the same name. If a name does not match, the
blob is simply ignored! This gives a powerful mechanic for partially loading
blobs, but be careful when remaining your layers!

Training a network
~~~~~~~~~~~~~~~~~~

To train a network, you can use the `scikit-learn` like method
:py:func:`barrista.net.Net.fit`. It is very powerful and can be used in many
different ways! While maintaining nearly all configurability of the caffe
solvers, it adds callback functionality and is a lot easier to use.

The only required method parameter is the number of iterations that you want
to train your network with. If you configured it with data-layers that are
loading data from external sources, you just have to decide about the kind
of solver to use and probably specify its learning rate. For this example,
we use in-memory data from Python for the training, and some monitors to
generate outputs::

    from barrista import solver
    from barrista.monitoring import ProgressIndicator, Checkpointer

    X = np.zeros((11, 3, 51, 51), dtype='float32')
    Y = np.ones((11, 1), dtype='float32')

    # Configure our monitors.
    progress = ProgressIndicator()
    checkptr = Checkpointer('test_net_', 50)
    # Run the training.
    net.fit(100,
            solver.SGDSolver(base_lr=0.01, snapshot_prefix='test_net_'),
            {'data': X,  # 'data' and 'annotations' are the input layer names.
             'annotations': Y}, # optional (if you have, e.g., a DataLayer)
            test_interval=50,  # optional
            X_val={'data': X,  # optional
                   'annotations': Y},
            after_batch_callbacks=[progress, checkptr],  # optional
            after_test_callbacks=[progress])  # optional

The parameters ``test_interval``, ``X_val`` and ``Y_val`` are optional. If they
are specified, there is a test performed on the validation set in
regular intervals.

Note that all iteration parameters are speaking of 'true' iterations, i.e.,
not batch iterations but sample iterations. This is, why they must be a
multiple of the batch size (e.g., for a network with a batch size of 10,
you have to do at least 10 training iterations, and one batch will be
used for the training).

The :py:class:`barrista.monitoring.Checkpointer` is used to write the network
blobs to a file, which can be loaded later using the function
:py:func:`barrista.net.Net.load_blobs_from` as well as the respective
solverstate. The ``snapshot_prefix`` provided to the solver and the
checkpointer prefix must match for this to work correctly.

Getting predictions
~~~~~~~~~~~~~~~~~~~

In the spirit of the `scikit-learn` library, we added the method
:py:func:`barrista.net.Net.predict` to get predictions for you, while
maintaining a clear separation of data preprocessing:

* It is YOUR responsibility to prepare the data in an iterable object
  of numpy arrays with the correctly matching first dimension (i.e.,
  the number of channels).
* The method will match the data to the input size of the network and
  forward propagate it in batches.

By default, it rescales the examples using
bicubic interpolation to the full input field size of the network, but if you
set ``pad_instead_of_rescale``, they will be instead padded to be centered in
the input field. If you choose padding and ``return_unprocessed_outputs`` is
set to ``False``, the data will automatically be reduced to the relevant
area.

You may
optionally set callback functions in between the batches to, e.g.,
update progress indicators::

    from barrista.monitoring import ProgressIndicator
    # Only the number of channels (3) must match.
    inputs = np.zeros((20, 3, 10, 10))
    results = net.predict(inputs,
                          after_batch_callbacks=[ProgressIndicator()])
    # This works for single-input networks. If you have multiple inputs, just
    # provide a dicitonary of layer-names with arrays, as for the fit-method.
    # Similarly, in case of a single-output network, this method returns a
    # single list of predictions, or, in case of a multi-output network,
    # a dictionary of output layer names with their respective output lists.
    print(results)

========================================================
Using different architectures to ``fit`` and ``predict``
========================================================

You have many possibilities to condition the network layout for the very same
network depending on it's state. It has
:py:attr:`barrista.design.NetSpecification.phase`,
:py:attr:`barrista.design.NetSpecification.level` and
:py:attr:`barrista.design.NetSpecification.stages`. The ``phase`` is used
to configure the net during the 'fit' progress to alternate between training
and validation sets. We offer a simple way of using the ``stages`` to switch
between different architectures for 'fit' and 'predict'.

When designing a network, you can specify the optional parameters
``predict_inputs`` and ``predict_input_shapes``. If you do so, when
instantiating the net, a second version of the net with the stages set only
to ``predict`` is created (with shared weights with the main network) and
automatically used when calling the :py:func:`barrista.net.Net.predict`
method (for an illustration of this behavior, see also the documentation for
:py:class:`barrista.design.NetSpecification`).
This is a very convenient way of using your networks comfortably and
just as expected, while maintaining a high level of convenience::

    netspec = design.NetSpecification([[10, 3, 51, 51], [10]],
                                      inputs=['data', 'annotations'],
                                      predict_inputs=['data'],
                                      predict_input_shapes=[[10, 3, 51, 51]])
    # ... add layers as usual.
    # This is the last regular one. Use `tops` to give its outputs a
    # simple-to-remember name.
    layers.append(InnerProductLayer(tops=['net_out'], InnerProduct_num_output=10))
    # Add a layer for being used by the `predict` method:
    layers.append(SoftmaxLayer(bottoms=['net_out'],
                               tops=['out'],
                               include_stages=['predict']))
    # Add layers for being used by the `fit` method:
    layers.append(SoftmaxWithLossLayer(bottoms=['net_out', 'annotations'],
                                       include_stages=['fit']))
    layers.append(AccuracyLayer(name='accuracy',
                                bottoms=['net_out', 'annotations'],
                                include_stages=['fit']))

Remember that you can additionally use any other conditional criteria such as
``phase`` and ``level`` to further customize the net.

Once instantiated, this net will output loss and accuracy when it's
:py:func:`barrista.net.Net.fit`
method is called, and output softmaxed values when it's
:py:func:`barrista.net.Net.predict` method is called. You can find an example
for this in the file ``barrista/examples/showcase.py``.
