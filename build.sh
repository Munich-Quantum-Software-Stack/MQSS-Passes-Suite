#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# TODO 
bash qdmi/build.sh

# Install the dependencies
sudo apt install -y cmake llvm rabbitmq-server g++ || true
if [ -e /usr/local/lib/librabbitmq.so ]; then
    echo "RabbitMQ-C is already installed. Skipping installation."
else
    curl -LO https://github.com/alanxz/rabbitmq-c/archive/refs/tags/v0.13.0.tar.gz
    tar -xf v0.13.0.tar.gz
    cd rabbitmq-c-0.13.0/
    mkdir build/ 2> /dev/null || true
    cd build/
    cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DENABLE_SSL_SUPPORT=OFF ..
    sudo cmake --build . --target install
    sudo ldconfig
    cd ../..
    rm -rf rabbitmq-c-0.13.0/
fi

# Build the pass runner
mkdir build/ 2> /dev/null || true
cd build/
export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
cmake -DCMAKE_INSTALL_PREFIX=$HOME -DCUSTOM_QDMI_PATH=qdmi ..
cmake --build .
#sudo make install

# Run the pass runner
#"$HOME/bin/daemon_d" log $HOME # usage: daemon_d [screen|log PATH]
./daemon_d log $HOME # usage: daemon_d [screen|log PATH]
