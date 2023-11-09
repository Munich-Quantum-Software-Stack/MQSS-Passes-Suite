#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# TODO 
bash qdmi/build.sh

# Install the dependencies
sudo apt install -y cmake llvm libopenmpi-dev g++ || true

# Build the pass runner
mkdir build/ 2> /dev/null || true
cd build/
export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
cmake -DCMAKE_INSTALL_PREFIX=$HOME -DCUSTOM_QDMI_PATH=qdmi ..
cmake --build .
#sudo make install

# Run the pass runner
#"$HOME/bin/daemon_d" log $HOME # usage: daemon_d [screen|log PATH]
#./daemon_d log $HOME # usage: daemon_d [screen|log PATH]
