#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Install the dependencies
sudo apt install -y cmake || true

# Build the pass selector runner
mkdir build/ 2> /dev/null || true
cd build/
cmake -DCUSTOM_EXECUTABLE_NAME=qselectorrunner_d ..
cmake --build .

export LC_ALL=en_US.UTF-8

# Run the pass selector runner
./qselectorrunner_d

