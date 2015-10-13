# -*- coding: utf-8 -*-
"""Implements some useful tools."""
# pylint: disable=C0103
from __future__ import print_function


import warnings as _warnings
import os as _os
import sys as _sys

import numpy as np
from tempfile import mkdtemp


def pbufToPyEnum(pbufenum):
    r"""Helper function to create a Python enum out of a protobuf one."""
    enums = dict(list(pbufenum.items()))
    return type('Enum', (), enums)


def chunks(seq, size):
    r"""
    Create chunks of ``size`` of ``seq``.

    See http://stackoverflow.com/questions/434287/
    what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks.
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def pad(image, input_dims, get_padding=False, val=0, pad_at_least=False):
    r"""
    Pad an image with given scale to the appropriate dimensions.

    The scaled image must fit into the input_dims, otherwise an exception is
    thrown!

    :param image: 3D numpy array.
      The image to pad, as (C, H, W).

    :param input_dims: tuple(int).
      A two-tuple of ints with the first value specifying height, the second
      width.

    :param get_padding: bool.
      If set to True, returns a second value, which is a tuple of two-tuples,
      where each tuple contains the two values for left-right paddings for
      one of the image dimensions.

    :param val: float.
      The value to pad with.

    :param pad_at_least: bool.
      If set to True, allows input_dims that are smaller than the image size.
      Otherwise, it throws in this case.

    :returns: 3D array, padded or, if ``get_padding``,
      (3D array, tuple(two-tuples)).
    """
    assert len(input_dims) == 2
    assert image.ndim == 3
    IMAGE_DIMS = np.array(image.shape[1:])
    SCALED_DIMS = IMAGE_DIMS[:].astype('int')
    WORK_SCALE_IMAGE = image
    PAD_WIDTH = (input_dims[1] - SCALED_DIMS[1]) / 2.0
    PAD_HEIGHT = (input_dims[0] - SCALED_DIMS[0]) / 2.0
    if not pad_at_least:
        assert PAD_WIDTH >= 0. and PAD_HEIGHT >= 0.
    else:
        PAD_WIDTH = max(0, PAD_WIDTH)
        PAD_HEIGHT = max(0, PAD_HEIGHT)
    # Padding is done, e.g., in deeplab, first with the mean values,
    # only to subtract the mean of the entire image, resulting in
    # 0. values in the padded areas. We're doing that here directly.
    padding = ((0, 0),
               (int(np.floor(PAD_HEIGHT)), int(np.ceil(PAD_HEIGHT))),
               (int(np.floor(PAD_WIDTH)), int(np.ceil(PAD_WIDTH))))
    padded = np.pad(WORK_SCALE_IMAGE, padding, 'constant', constant_values=val)
    if get_padding:
        return padded, padding
    else:
        return padded


class TemporaryDirectory(object):  # pylint: disable=R0903

    """Create and return a temporary directory.

    This has the same behavior as mkdtemp but can be used as a context manager.
    For example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.

    Source:
    http://stackoverflow.com/questions/19296146/
    tempfile-temporarydirectory-context-manager-in-python-2-7.
    """

    # pylint: disable=W0622
    def __init__(self, suffix="", prefix="tmp", dir=None):
        """Same parameters as ``mkdtemp``."""
        self._closed = False
        self.name = None  # Handle mkdtemp raising an exception
        self.name = mkdtemp(suffix, prefix, dir)

    def __repr__(self):
        """Plain string representation."""
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        """When entering the context."""
        return self.name

    def cleanup(self, _warn=False):
        """Guarantee a cleaned up state."""
        if self.name and not self._closed:
            try:
                self._rmtree(self.name)
            except (TypeError, AttributeError) as ex:
                # Issue #10188: Emit a warning on stderr
                # if the directory could not be cleaned
                # up due to missing globals
                if "None" not in str(ex):
                    raise
                print("ERROR: {!r} while cleaning up {!r}".format(ex, self,),
                      file=_sys.stderr)
                return
            self._closed = True
            if _warn:
                self._warn("Implicitly cleaning up {!r}".format(self),
                           self._warn.ResourceWarning)

    def __exit__(self, exc, value, tb):
        """On leaving the context."""
        self.cleanup()

    def __del__(self):
        """On deleting the context."""
        # Issue a ResourceWarning if implicit cleanup needed.
        self.cleanup(_warn=True)

    # The following code attempts to make
    # this class tolerant of the module nulling out process
    # that happens during CPython interpreter shutdown
    # Alas, it doesn't actually manage it. See issue #10188.
    _listdir = staticmethod(_os.listdir)
    _path_join = staticmethod(_os.path.join)
    _isdir = staticmethod(_os.path.isdir)
    _islink = staticmethod(_os.path.islink)
    _remove = staticmethod(_os.remove)
    _rmdir = staticmethod(_os.rmdir)
    _warn = _warnings.warn

    def _rmtree(self, path):
        """
        Essentially a stripped down version of shutil.rmtree.

        We can't use globals because they may be None'ed out at shutdown.
        """
        for name in self._listdir(path):
            fullname = self._path_join(path, name)
            try:
                isdir = self._isdir(fullname) and not self._islink(fullname)
            except OSError:
                isdir = False
            if isdir:
                self._rmtree(fullname)
            else:
                try:
                    self._remove(fullname)
                except OSError:
                    pass
        try:
            self._rmdir(path)
        except OSError:
            pass
