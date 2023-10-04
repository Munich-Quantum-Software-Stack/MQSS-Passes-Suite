#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Build the selector
rm selector 2> /dev/null || true
mpic++ -std=c++14 selector.cpp -o selector

# Run the selector
./selector

