#!/bin/bash

# Set installation path and location of the interface
export INSTALL_PREFIX=$HOME
export QDMI_PATH=$PWD/qdmi

# Set the script to exit on any non-zero status
set -e

# Compile the chosen interface
bash $QDMI_PATH/build.sh

# Install the dependencies
sudo apt install -y cmake llvm rabbitmq-server g++ || true
if [ -e /usr/local/lib/librabbitmq.so ]; then
    echo "RabbitMQ-C is already installed. Skipping installation."
else
    curl -LO https://github.com/alanxz/rabbitmq-c/archive/refs/tags/v0.13.0.tar.gz
    tar -xf v0.13.0.tar.gz
    cd rabbitmq-c-0.13.0/
    mkdir -p build
    cd build
    cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DENABLE_SSL_SUPPORT=OFF ..
    sudo cmake --build . --target install
    sudo ldconfig
    cd ../..
    rm -rf rabbitmq-c-0.13.0/
fi

# Configure RabbitMQ
hosts_file="/etc/hosts"
hostname_entry="127.0.0.1 rabbitmq"
if grep -qF "$hostname_entry" "$hosts_file"; then
    echo "RabbitMQ is already configured in this system."
else
    echo "$hostname_entry" | cat - "$hosts_file" > temp && sudo mv -f temp "$hosts_file"
fi

# Build the pass runner
mkdir -p build
cd build
#
export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
export LD_LIBRARY_PATH=$QDMI_PATH/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/pass_runner:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/pass_runner/passes:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/selector_runner:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/selector_runner/selectors:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/scheduler_runner:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/scheduler_runner/schedulers:$LD_LIBRARY_PATH
#
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DCUSTOM_QDMI_PATH=$QDMI_PATH ..
sudo cmake --build . --target install
sudo ldconfig

# Run the pass runner
daemon_d log $INSTALL_PREFIX # usage: daemon_d [screen|log PATH]

