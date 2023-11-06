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
cmake -DCUSTOM_EXECUTABLE_NAME=qpassrunner_d -DCUSTOM_QDMI_PATH=qdmi ..
cmake --build .
sudo make install

#sudo cmake -P -E copy ${CMAKE_SOURCE_DIR}/benchmarks/test.ll /usr/local/bin/src/schedulers/benchmarks/
sudo cp benchmarks/test.ll  /usr/local/bin/src/schedulers/benchmarks/

#export LC_ALL=en_US.UTF-8

# Run the pass runner
qpassrunner_d log # usage: qpassrunner_d [screen|log]

