#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Build the test
g++ test.cpp connection_handling.cpp -o test -lrabbitmq

# Run the test
./test
