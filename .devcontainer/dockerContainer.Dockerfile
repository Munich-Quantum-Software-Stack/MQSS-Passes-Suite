FROM ghcr.io/nvidia/cuda-quantum-devdeps:ext-cu12.0-gcc11-main

ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

ARG workspace=.
ARG destination="$CUDAQ_REPO_ROOT"
ADD "$workspace" "$destination"
WORKDIR "$destination"

RUN git clone https://github.com/NVIDIA/cuda-quantum.git ${CUDAQ_REPO_ROOT}
ENV LLVM_PROJECTS="clang;lld;mlir;python-bindings;runtimes"
RUN mkdir -p /opt/nvidia/cuquantum /opt/nvidia/cutensor
ENV CUQUANTUM_INSTALL_PREFIX=/opt/nvidia/cuquantum
ENV PATH="$CUQUANTUM_INSTALL_PREFIX/bin:${PATH}"
ENV CUTENSOR_INSTALL_PREFIX=/opt/nvidia/cutensor
ENV PATH="$CUTENSOR_INSTALL_PREFIX/bin:${PATH}"
ENV LLVM_INSTALL_PREFIX=/opt/llvm
ENV PATH="$LLVM_INSTALL_PREFIX/bin:${PATH}"

RUN source ${CUDAQ_REPO_ROOT}/scripts/build_cudaq.sh -j 6

RUN apt-get update && \
    apt-get install -y --no-install-recommends libboost-program-options-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
