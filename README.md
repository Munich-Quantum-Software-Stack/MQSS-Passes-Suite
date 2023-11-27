# Quantum Resource Manager (QRM)

The entry point of the Quantum Resource Manager for selecting and applying LLVM passes to a Quantum Circuit described on a Quantum Intermediate Representation ([QIR](https://www.qir-alliance.org/projects/)) is `daemon_d`. This README provides instructions for installing and uninstalling `daemon_d`, as well as for running an example submitting a quantum task<!--Not to be confused with a qcommon QuantumTask-->.

## Downloading

1. To clone this repository to your local machine, use the following command:
   ```bash
   git clone https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes.git
   ```

2. After cloning, make sure you are the right branch:
   ```bash
   cd qir_passes
   git checkout NoSockets
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

To install the Quantum Resource Manager daemon system wide, follow these steps:

1. Install the required dependencies:
   ```bash
   sudo apt update
   sudo apt install -y cmake llvm rabbitmq-server g++ curl nlohmann-json3-dev
   ```

2. Navigate to the `qir_passes` directory (if you are not already there):
   ```bash
   cd qir_passes/
   ```

3. Run `make` to install `daemon_d`:
   - One can install the daemon in the default directory, i.e., `$HOME/bin`, with the following command:
      ```bash
      make FOMAC_PATH=fomac BACKENDS_PATH=backends install
      ```

   - Besides the directories with the chosen Figure of Merits and Constraints library (FoMaC) and the available backends, one may also specify the installation path, and a directory where the build files can be written to. Note that the equivalent command to the one above is:
      ```bash
      make INSTALL_PATH=$HOME FOMAC_PATH=fomac BACKENDS_PATH=backends BUILD_DIR=build install
      ```

## Uninstallation

If you ever need to uninstall `daemon_d`, follow these steps:

1. Navigate to the `qir_passes` directory (if you are not already there):
   ```bash
   cd qir_passes/
   ```

2. Run the uninstall target using sudo:
   ```bash
   sudo make uninstall
   ```

This will remove `daemon_d` from your system.

## Project Structure

The project structure is the following:

```
├─ .clang-format
├─ .gitignore
├─ .gitlab-ci.yml
├─ .pre-commit-config.yaml
├─ CMakeLists.txt
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md
├─ LICENSE
├─ Makefile
├─ README.md
├─ backends
│  ├─ CMakeLists.txt
│  ├─ JobRunner.hpp
│  ├─ IQMBackend.cpp
│  ├─ IQMBackend.hpp
│  ├─ Q5Backend.hpp
│  └─ Q20Backend.hpp
├─ benchmarks
│  └─ test.ll
├─ cmake
│  └─ FindSphinx.cmake
├─ docs
│  ├─ CMakeLists.txt
│  ├─ Doxyfile.in
│  └─ html
│     ├─ index.html
│     └─ ...
├─ flowcharts
│  ├─ flow.drawio
│  └─ flow.png
├─ fomac
│  ├─ CMakeLists.txt
│  ├─ fomac.cpp
│  └─ fomac.hpp
├─ scripts
│  ├─ kill_daemons.sh
│  └─ generate_docs.sh
├─ src
│  ├─ connection_handling.cpp
│  ├─ connection_handling.hpp
│  ├─ daemon_d.cpp
│  ├─ pass_runner
│  │  ├─ headers
│  │  │  ├─ llvm.hpp
│  │  │  └─ ...
│  │  ├─ PassRunner.cpp
│  │  └─ passes
│  │     ├─ CMakeLists.txt
│  │     └─ ...
│  ├─ qdmi
│  │  ├─ qdmi.cpp
│  │  └─ qdmi.hpp
│  ├─ scheduler_runner
│  │  ├─ schedulers
│  │  │  ├─ CMakeLists.txt
│  │  │  ├─ scheduler_round_robin.cpp
│  │  │  └─ ...
│  │  ├─ SchedulerRunner.cpp
│  │  └─ SchedulerRunner.hpp
│  └─ selector_runner
│     ├─ selectors
│     │  ├─ CMakeLists.txt
│     │  ├─ selector_all.cpp
│     │  └─ ...
│     ├─ SelectorRunner.cpp
│     └─ SelectorRunner.hpp
└─ tests
   └─ test.cpp
```

## Documentation and Resources

This section provides links to project documentation and additional resources:

- [Documentation](https://lrz-qct-qis.gitlabpages.devweb.mwn.de/quantum_intermediate_representation/qir_passes/files.html): Detailed documentation about the Quantum Resource Manager.
- [Wiki](https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/wikis/home): Project wiki with additional information and guides.
- [Contributing Guidelines](CONTRIBUTING.md): Document to understand the process for contributing to our project.
<!--
- Flowchart of the QIR Pass Runner daemon:
![Alt](flowcharts/flow.png)
-->

## Building Documentation

You can build the Quantum Resource Manager and generate its documentation locally using Doxygen:

1. Install the required dependencies for Doxygen:
   ```bash
   sudo apt update
   sudo apt install -y cmake llvm rabbitmq-server g++ flex bison
   ```

2. Run make:
   - One can install the daemon in the default directory, i.e., `$HOME/bin`, and generate its documentation with the following command:
      ```bash
      make FOMAC_PATH=fomac BACKENDS_PATH=backends docs
      ```

   - Besides the directory with the chosen Figures of Merit and Constraints library (FOMAC), one may also specify the installation path, and a directory where the build files can be written to. Note that the equivalent command to the one above is:
      ```bash
      make INSTALL_PATH=$HOME FOMAC_PATH=fomac BACKENDS_PATH=backends BUILD_DIR=build docs
      ```

3. Open the generated documentation in a web browser:
   ```bash
   xdg-open docs/html/index.html
   ```

   Alternatively, you can manually open the file `documentation/html/index.html` with your preferred web browser.

4. Once the forked branch is merged, the up-to-date documentation can be accessed online [here](https://lrz-qct-qis.gitlabpages.devweb.mwn.de/quantum_intermediate_representation/qir_passes/index.html).

## Running Examples

You can run the Quantum Resource Manager daemon and a test client as follows:

1. Navigate to the `qir_passes` directory (if you are not already there):
   ```bash
   cd qir_passes/
   ```

2. Run the following command:
   ```bash
   make test
   ```
