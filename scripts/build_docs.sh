#!/bin/bash

# Create the build directory
BUILD_DIR=docs/build
mkdir -p "$BUILD_DIR"

# Run CMake to generate the documentation
cmake -B "$BUILD_DIR" -S docs -Wno-dev -DBUILD_DOCS_STANDALONE=ON -DCMAKE_MODULE_PATH=../cmake

# Build the documentation
make -C "$BUILD_DIR" docs

# Stage the changes
git add docs/html
