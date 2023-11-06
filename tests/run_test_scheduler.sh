#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Build the test
g++ test_scheduler.cpp connection_handling.cpp -o test_scheduler -lrabbitmq

# Run the test
./test_scheduler
