#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Build the test
mpic++ -std=c++14 test_selector.cpp -o test_selector

# Run the test
./test_selector
