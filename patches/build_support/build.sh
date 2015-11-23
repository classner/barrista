#!/bin/bash
# Script called by Travis to build and test Caffe.
# Travis CI tests are CPU-only for lack of compatible hardware.

set -e
MAKE="make --jobs=$NUM_THREADS --keep-going"

mkdir build
cd build
CPU_ONLY=" -DCPU_ONLY=ON"
if ! $WITH_CUDA; then
  CPU_ONLY=" -DCPU_ONLY=OFF"
fi
PYTHON_ARGS=""
if [ "$PYTHON_VERSION" = "3" ]; then
  PYTHON_ARGS="$PYTHON_ARGS -Dpython_version=3 -DBOOST_LIBRARYDIR=$CONDA_DIR/lib/"
else
  PYTHON_ARGS="$PYTHON_ARGS -Dpython_version=2 -DBOOST_LIBRARYDIR=$CONDA_DIR/lib/ -DPYTHON_EXECUTABLE=$CONDA_DIR/bin/python -DPYTHON_INCLUDE_DIR=$CONDA_DIR/include/python2.7/ -DPYTHON_LIBRARY=$CONDA_DIR/lib/libpython2.7.so"
fi
if $WITH_IO; then
  IO_ARGS="-DUSE_OPENCV=ON -DUSE_LMDB=ON -DUSE_LEVELDB=ON"
else
  IO_ARGS="-DUSE_OPENCV=OFF -DUSE_LMDB=OFF -DUSE_LEVELDB=OFF"
fi
cmake -DBUILD_python=ON -DCMAKE_BUILD_TYPE=Release $CPU_ONLY $PYTHON_ARGS -DCMAKE_INCLUDE_PATH="$CONDA_DIR/include/" -DCMAKE_LIBRARY_PATH="$CONDA_DIR/lib/" $IO_ARGS ..
$MAKE
$MAKE install
cd -
