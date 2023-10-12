#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Build the test
mpic++ -std=c++14 test.cpp -o test

# Run the test
./test
