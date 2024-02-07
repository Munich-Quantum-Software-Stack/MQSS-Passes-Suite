# QIR Passes

This project focuses on developing custom LLVM-compliant passes designed to optimize quantum circuits described in QIR (Quantum Intermediate Representation). Leveraging LLVM's powerful infrastructure, these passes are tailored to enhance the performance and efficiency of quantum computations, offering a suite of optimization tools specifically crafted for QIR-based quantum circuits.

## Downloading

1. To clone this repository to your local machine, use the following command:
   ```bash
   git clone https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes.git
   ```

2. After cloning, make sure you are at the right branch:
   ```bash
   cd qir_passes
   git checkout passes
   ```

3. Install `pre-commit` using `pip` (Python's package manager) and other required dependencies using `apt`:
   ```bash
   sudo apt update
   sudo apt install -y cmake clang-format pre-commit cmakelang python3-pip
   pip3 install cmake_format
   echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
   source ~/.bashrc
   ```

4. After installing `pre-commit`, set up the hooks specified in `.pre-commit-config.yaml`:
   ```bash
   pre-commit install
   make pre-commit
   ```

## Building

To install the QIR Passes system wide, follow these steps:

1. Install the required dependencies:
   ```bash
   sudo apt update
   sudo apt install -y cmake llvm rabbitmq-server g++ curl libgtest-dev nlohmann-json3-dev
   ```

2. Install FoMaC, QDMI, and the Backends libraries:
   - Clone the project:
      ```bash
      git clone https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes.git
      ```
   - Navigate to the `quantum-resource-manager` directory (if you are not already there) and move to the right branch:
      ```bash
      cd quantum-resource-manager
      git checkout develop
      ```

   - Build and install the libraries:
      ```bash
      export INSTALLATION_PATH=installation/path/libraries

      make INSTALL_PATH=$INSTALLATION_PATH \
           QDMI_PATH=qdmi \
           FOMAC_PATH=fomac \
           BACKENDS_PATH=backends \
           BUILD_DIR=build \
           install
      ```

3. Navigate to the `qir_passes` directory (if you are not already there):
   ```bash
   cd qir_passes/
   ```

4. Run `make` to install the passes as shared libraries and add the installation path to the `PASSES` environment variable:
   - One can install the QIR Passes with the following command:
      ```bash
      export QDMI_INCLUDE_PATH=path/to/qdmi
      export FOMAC_INCLUDE_PATH=path/to/fomac
      export BACKENDS_INCLUDE_PATH=path/to/backends
      export INSTALLATION_PATH=installation/path/passes

      make INSTALL_PATH=$INSTALLATION_PATH \
           QDMI_INCLUDE_PATH=$QDMI_INCLUDE_PATH \
           FOMAC_INCLUDE_PATH=$FOMAC_INCLUDE_PATH \
           BACKENDS_INCLUDE_PATH=$BACKENDS_INCLUDE_PATH \
           install

      export PASSES=$PASSES:$INSTALLATION_PATH
      ```

   - Besides the directory with the chosen Figures of Merit and Constraints library (FOMAC), the Quantun Device Management Interface (QDMI), the backends, and the installation path, one may also specify the directory where the build files can be written to. Note that the equivalent command to the one above is:
      ```bash
      export QDMI_INCLUDE_PATH=path/to/qdmi
      export FOMAC_INCLUDE_PATH=path/to/fomac
      export BACKENDS_INCLUDE_PATH=path/to/backends
      export INSTALLATION_PATH=installation/path/passes

      make INSTALL_PATH=$INSTALLATION_PATH \
           BUILD_DIR=build \
           QDMI_INCLUDE_PATH=$QDMI_INCLUDE_PATH \
           FOMAC_INCLUDE_PATH=$FOMAC_INCLUDE_PATH \
           BACKENDS_INCLUDE_PATH=$BACKENDS_INCLUDE_PATH \
           install

      export PASSES=$PASSES:$INSTALLATION_PATH
      ```

## Uninstallation

If you ever need to uninstall the passes, follow these steps:

1. Navigate to the `qir_passes` directory (if you are not already there):
   ```bash
   cd qir_passes/
   ```

2. Run the uninstall target using sudo:
   ```bash
   sudo make uninstall
   ```

This will remove the passes from your system.

## Project Structure

The project structure is the following:

```
├─ .clang-format
├─ .gitignore
├─ .gitlab
│  ├─ issue_templates
│  │  └─ new_issue.md
│  └─ merge_request_templates
│     └─ new_merge_request.md
├─ .gitlab-ci.yml
├─ .pre-commit-config.yaml
├─ CMakeLists.txt
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md
├─ LICENSE
├─ Makefile
├─ README.md
├─ cmake
│  └─ FindSphinx.cmake
├─ docs
│  ├─ CMakeLists.txt
│  ├─ Doxyfile.in
│  └─ html
│     ├─ index.html
│     └─ ...
└─ src
   ├─ headers
   │  ├─ llvm.hpp
   │  ├─ utilities.hpp
   │  └─ ...
   └─ passes
      ├─ utilities.cpp
      └─ ...
```

## Documentation and Resources

This section provides links to project documentation and additional resources:

- [Documentation](https://lrz-qct-qis.gitlabpages.devweb.mwn.de/quantum_intermediate_representation/qir_passes/files.html): Detailed documentation about the Quantum Resource Manager.
- [Wiki](https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/wikis/home): Project wiki with additional information and guides.
- [Contributing Guidelines](CONTRIBUTING.md): Document to understand the process for contributing to our project.

## Building Documentation

You can build the Quantum Resource Manager and generate its documentation locally using Doxygen:

1. Install the required dependencies for Doxygen:
   ```bash
   sudo apt update
   sudo apt install -y cmake llvm rabbitmq-server g++ curl flex bison libgtest-dev nlohmann-json3-dev
   ```

2. Run make:
   - One can install the QIR Passes and generate their documentation with the following command:
      ```bash
      export QDMI_INCLUDE_PATH=path/to/qdmi
      export FOMAC_INCLUDE_PATH=path/to/fomac
      export BACKENDS_INCLUDE_PATH=path/to/backends
      export INSTALLATION_PATH=installation/path/passes

      make INSTALL_PATH=$INSTALLATION_PATH \
           QDMI_INCLUDE_PATH=$QDMI_INCLUDE_PATH \
           FOMAC_INCLUDE_PATH=$FOMAC_INCLUDE_PATH \
           BACKENDS_INCLUDE_PATH=$BACKENDS_INCLUDE_PATH \
           docs

      export PASSES=$PASSES:$INSTALLATION_PATH
      ```

   - Besides the directory with the chosen Figures of Merit and Constraints library (FOMAC), the Quantun Device Management Interface (QDMI), the backends, and the installation path, one may also specify the directory where the build files can be written to. Note that the equivalent command to the one above is:
      ```bash
      export QDMI_INCLUDE_PATH=path/to/qdmi
      export FOMAC_INCLUDE_PATH=path/to/fomac
      export BACKENDS_INCLUDE_PATH=path/to/backends
      export INSTALLATION_PATH=installation/path/passes

      make INSTALL_PATH=$INSTALLATION_PATH \
           BUILD_DIR=build \
           QDMI_INCLUDE_PATH=$QDMI_INCLUDE_PATH \
           FOMAC_INCLUDE_PATH=$FOMAC_INCLUDE_PATH \
           BACKENDS_INCLUDE_PATH=$BACKENDS_INCLUDE_PATH \
           docs

      export PASSES=$PASSES:$INSTALLATION_PATH
      ```

3. Open the generated documentation in a web browser:
   ```bash
   xdg-open docs/html/index.html
   ```

   Alternatively, you can manually open the file `documentation/html/index.html` with your preferred web browser.

4. Once the forked branch is merged, the up-to-date documentation can be accessed online [here](https://lrz-qct-qis.gitlabpages.devweb.mwn.de/quantum_intermediate_representation/qir_passes/index.html).
