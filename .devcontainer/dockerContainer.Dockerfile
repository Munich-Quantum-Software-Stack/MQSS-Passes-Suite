# Use NVIDIA's CUDA Quantum base image
# Use NVIDIA's CUDA Quantum base image
FROM ghcr.io/nvidia/cuda-quantum-devdeps:ext-cu12.0-gcc11-main

ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"
ENV LLVM_INSTALL_PREFIX=/opt/llvm
ENV CUQUANTUM_INSTALL_PREFIX=/opt/nvidia/cuquantum
ENV CUTENSOR_INSTALL_PREFIX=/opt/nvidia/cutensor

# Clone the repository
RUN git clone https://github.com/NVIDIA/cuda-quantum.git ${CUDAQ_REPO_ROOT}

# Add local files to a separate directory (if needed)
ARG workspace=.
ARG destination="/workspaces/custom"
ADD "$workspace" "$destination"

# Install prerequisites
WORKDIR "$CUDAQ_REPO_ROOT"

ENV MAKEFLAGS="-j7"
ENV NINJAFLAGS="-j7"

RUN LLVM_PROJECTS="clang;lld;mlir;python-bindings;runtimes" source "$CUDAQ_REPO_ROOT"/scripts/install_prerequisites.sh -t clang16

# Set terminal type for colors
ENV TERM xterm-256color
# Install color-supporting packages
RUN apt update && apt install -y bash coreutils git
# Enable colored ls
RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc
# Enable Git colors
RUN git config --global color.ui auto
CMD ["bash"]
