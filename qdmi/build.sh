#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Install the dependencies
sudo apt install -y cmake || true

# Build qdmi
cd qdmi
mkdir build/ 2> /dev/null || true
cd build
cmake ..
make
cd ../..

