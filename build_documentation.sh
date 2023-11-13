#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Install the dependencies
sudo apt install -y flex bison || true

# Check if doxygen is installed
if command -v "doxygen"; then
    echo "Doxygen is already installed"
else
    echo "Installing Doxygen"
    git clone https://github.com/doxygen/doxygen.git
    cd doxygen
    mkdir build/ 2> /dev/null || true
    cd build
    cmake -G "Unix Makefiles" ..
    make
    sudo make install
    cd ../..
    rm -rf doxygen
fi

doxygen Doxyfile

