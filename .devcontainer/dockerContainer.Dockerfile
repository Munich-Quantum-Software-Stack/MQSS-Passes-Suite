# Use NVIDIA's CUDA Quantum base image
# [Operating System]
ARG base_image=ubuntu:22.04

FROM ghcr.io/nvidia/cuda-quantum-devdeps:ext-cu12.0-gcc11-main

ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

ARG workspace=.
ARG destination="$CUDAQ_REPO_ROOT"
ADD "$workspace" "$destination"
WORKDIR "$destination"

#RUN git clone https://github.com/NVIDIA/cuda-quantum.git ${CUDAQ_REPO_ROOT}

# Install Boost development libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends ibverbs-providers=39.0-1 libibverbs1=39.0-1 && \
    apt-get install -y --no-install-recommends libboost-all-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#RUN LLVM_PROJECTS="clang;lld;mlir;python-bindings;runtimes" source ${CUDAQ_REPO_ROOT}/scripts/build_cudaq.sh -j 7

