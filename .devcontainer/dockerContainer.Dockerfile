# [Operating System]
ARG base_image=ubuntu:22.04

# [CUDA-Q Dependencies]
FROM ${base_image} AS prereqs
SHELL ["/bin/bash", "-c"]
ARG toolchain=gcc11

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

## [Prerequisites]
RUN apt-get update && apt-get install -y --no-install-recommends python3 && \
    apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

## [Environment Variables]
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
ENV CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
ENV LLVM_INSTALL_PREFIX=/usr/local/llvm
ENV BLAS_INSTALL_PREFIX=/usr/local/blas
ENV ZLIB_INSTALL_PREFIX=/usr/local/zlib
ENV OPENSSL_INSTALL_PREFIX=/usr/local/openssl
ENV CURL_INSTALL_PREFIX=/usr/local/curl
ENV AWS_INSTALL_PREFIX=/usr/local/aws

## [Build Dependencies]
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git unzip \
        python3-dev python3-pip && \
    python3 -m pip install --no-cache-dir numpy && \
    apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*
ADD scripts/install_toolchain.sh /cuda-quantum/scripts/install_toolchain.sh









#FROM ghcr.io/nvidia/cuda-quantum-devdeps:ext-cu12.0-gcc11-main
#
#ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
#ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
#ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
#ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"
#
#ARG workspace=.
#ARG destination="$CUDAQ_REPO_ROOT"
#ADD "$workspace" "$destination"
#WORKDIR "$destination"
#
##RUN git clone https://github.com/NVIDIA/cuda-quantum.git ${CUDAQ_REPO_ROOT}
##ENV LLVM_PROJECTS="clang;lld;mlir;python-bindings;runtimes;compiler-rt"
##RUN mkdir -p /opt/nvidia/cuquantum /opt/nvidia/cutensor
##ENV CUQUANTUM_INSTALL_PREFIX=/opt/nvidia/cuquantum
##ENV PATH="$CUQUANTUM_INSTALL_PREFIX/bin:${PATH}"
##ENV CUTENSOR_INSTALL_PREFIX=/opt/nvidia/cutensor
##ENV PATH="$CUTENSOR_INSTALL_PREFIX/bin:${PATH}"
##ENV LLVM_INSTALL_PREFIX=/opt/llvm
##ENV PATH="$LLVM_INSTALL_PREFIX/bin:${PATH}"
##
##RUN source ${CUDAQ_REPO_ROOT}/scripts/build_cudaq.sh -j 6
#
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends libboost-program-options-dev && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*
