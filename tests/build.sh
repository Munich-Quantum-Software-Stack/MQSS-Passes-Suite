#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Build the client
rm client 2> /dev/null || true
mpic++ -std=c++14 test.cpp -o test

# Run the client
./test
