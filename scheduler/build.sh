#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# TODO
cd ..
bash qdmi/build.sh
cd scheduler/

# Install the dependencies
sudo apt install -y cmake llvm rabbitmq-server || true
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

# Build the scheduler runner
mkdir build/ 2> /dev/null || true
cd build/
export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
cmake -DCUSTOM_QDMI_PATH=../qdmi ..
cmake --build .
sudo make install

export LC_ALL=en_US.UTF-8

# Run the scheduler runner
qschedulerrunner_d log # usage: qschedulerrunner_d [screen|log] 

