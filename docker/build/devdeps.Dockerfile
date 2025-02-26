# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the development environment that contains the necessary development 
# dependencies for building and testing CUDA-Q. This does not include the CUDA, OpenMPI 
# and other dependencies that some of the simulator backends require. These backends
# will be omitted from the build if this environment is used.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:${toolchain}-latest -f docker/build/devdeps.Dockerfile --build-arg toolchain=$toolchain .
#
# The variable $toolchain indicates which compiler toolchain to build the LLVM libraries with. 
# The toolchain used to build the LLVM binaries that CUDA-Q depends on must be used to build
# CUDA-Q. This image sets the CC and CXX environment variables to use that toolchain. 
# Currently, clang16, clang15, gcc12, and gcc11 are supported. To use a different 
# toolchain, add support for it to the install_toolchain.sh script. If the toolchain is set to llvm, 
# then the toolchain will be built from source.

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:ext-cu12.0-gcc11-main
FROM $base_image

# Ensure Git is installed
#RUN apt-get update && apt-get install -y git curl

#ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
#ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
#ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
#ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

# Clone the repository before adding the workspace
#RUN git clone --depth=1 https://github.com/NVIDIA/cuda-quantum.git "$CUDAQ_REPO_ROOT"

ARG workspace=.
#ARG destination="$CUDAQ_REPO_ROOT"
#ADD "$workspace" "$destination"
WORKDIR "$destination"

RUN git clone --depth=1 https://github.com/NVIDIA/cuda-quantum.git /workspaces/cuda-quantum    


