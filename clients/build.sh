#!/bin/bash

# Set the script to exit on any non-zero status
set -e

# Build the client
mpic++ -std=c++14 client.cpp -o client

# Run the client
./client
