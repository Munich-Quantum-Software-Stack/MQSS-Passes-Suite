# Use NVIDIA's CUDA Quantum base image
# Use NVIDIA's CUDA Quantum base image
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
RUN source ${CUDAQ_REPO_ROOT}/scripts/build_cudaq.sh -j 7
































