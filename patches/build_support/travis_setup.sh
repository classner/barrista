#!/bin/bash
# This script must be run with sudo.

set -e

MAKE="make --jobs=$NUM_THREADS"
# Install apt packages where the Ubuntu 12.04 default and ppa works for Caffe

# This ppa is for gflags and glog
add-apt-repository -y ppa:tuleu/precise-backports
apt-get -y update
apt-get install \
    wget git curl \
    python-dev python-numpy python3-dev\
    libleveldb-dev libsnappy-dev libopencv-dev \
    libatlas-dev libatlas-base-dev \
    libhdf5-serial-dev libgflags-dev libgoogle-glog-dev \
    bc

# Add a special apt-repository to install CMake 2.8.9 for CMake Caffe build,
# if needed.  By default, Aptitude in Ubuntu 12.04 installs CMake 2.8.7, but
# Caffe requires a minimum CMake version of 2.8.8.
if $WITH_CMAKE; then
  # cmake 3 will make sure that the python interpreter and libraries match
  wget --no-check-certificate http://www.cmake.org/files/v3.2/cmake-3.2.3-Linux-x86_64.sh -O cmake3.sh
  chmod +x cmake3.sh
  ./cmake3.sh --prefix=/usr/ --skip-license --exclude-subdir
fi

# Install LMDB
LMDB_URL=https://github.com/LMDB/lmdb/archive/LMDB_0.9.14.tar.gz
LMDB_FILE=/tmp/lmdb.tar.gz
pushd .
wget $LMDB_URL -O $LMDB_FILE
tar -C /tmp -xzvf $LMDB_FILE
cd /tmp/lmdb*/libraries/liblmdb/
$MAKE
$MAKE install
popd
rm -f $LMDB_FILE

# Install the Python runtime dependencies via miniconda (this is much faster
# than using pip for everything).
export PATH=$CONDA_DIR/bin:$PATH
#if [ ! -d $CONDA_DIR ]; then
  rm -rf $CONDA_DIR
	if [ "$PYTHON_VERSION" -eq "3" ]; then
		wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
	else
		wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
	fi
	chmod +x miniconda.sh
	./miniconda.sh -b -p $CONDA_DIR

	conda update --yes conda
  if [ "$PYTHON_VERSION" -eq "3" ]; then
    # The version of boost we're using for Python 3 depends on 3.4 for now.
    conda install --yes python=3.4
  fi
	conda install --yes numpy scipy matplotlib scikit-image pip
	# Let conda install boost (so that boost_python matches)
	conda install --yes -c https://conda.binstar.org/menpo boost=1.56.0
#fi

pushd .
wget https://github.com/google/protobuf/archive/3.0.0-GA.tar.gz -O protobuf-3.tar.gz
tar -C /tmp -xzvf protobuf-3.tar.gz
cd /tmp/protobuf-3*/
./autogen.sh
./configure --prefix=$CONDA_DIR
$MAKE
$MAKE install
popd

if [ "$PYTHON_VERSION" -eq "3" ]; then
	  pip install --pre protobuf
else
	  pip install --pre protobuf
    pip install coveralls
    pip install coverage
    pip install tqdm
fi
