#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Install the dependencies
sudo apt install -y flex bison || true

# Build the pass runner
git clone https://github.com/doxygen/doxygen.git
cd doxygen
mkdir build/ 2> /dev/null || true
cd build
cmake -G "Unix Makefiles" ..
make
sudo make install
cd ../..
doxygen Doxyfile

