#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Install the dependencies
sudo apt install -y cmake llvm libopenmpi-dev || true

# Build the QMPM
mkdir build/ 2> /dev/null || true
cd build/
export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
cmake -DCUSTOM_MAIN_SOURCE=src/QMPM_d.cpp -DCUSTOM_EXECUTABLE_NAME=QMPM_d ..
cmake --build .

# Run the QMPM
./QMPM_d

