#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Install the dependencies
sudo apt install -y cmake llvm libopenmpi-dev || true

# Build the pass runner
mkdir build/ 2> /dev/null || true
cd build/
export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
cmake -DCUSTOM_MAIN_SOURCE=src/qpassrunner_d.cpp -DCUSTOM_EXECUTABLE_NAME=qpassrunner_d ..
cmake --build .

# Run the pass runner
./qpassrunner_d

