# Development Guide

<!-- IMPORTANT: Keep the line above as the first line. -->
<!----------------------------------------------------------------------------
Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------- -->

<!-- This file is a static page and included in the CMakeLists.txt file. -->

Ready to contribute to the Collection of MLIR Passes of the MQSS? This guide will help you get
started.

## Installing LLVM/MLIR

Since the MQSS compiler is based on LLVM/MLIR infrastructure and CudaQ compiler, your system must
contain an installation of the LLVM project as a main prerequisite.

We recommend using the scripts given by CudaQ (see
<https://github.com/NVIDIA/cuda-quantum/tree/main/scripts>) . Those scripts help install LLVM/MLIR.

We recommend `clang16` as toolchain for the compilation of the LLVM project. Use the script
`install_toolchain.sh` as follows:

```sh
 bash scripts/install_toolchain.sh -t clang16
```

Then, you can install LLVM/MLIR and the prerequisites required by CudaQ by running:

```sh
 LLVM_PROJECTS="clang;lld;mlir;python-bindings;compiler-rt"  bash scripts/install_prerequisites.sh
```

\note Do not forget to include `compiler-rt` in the `LLVM_PROJECTS`. This is required by some of the
MQSS MLIR passes. If you do not include it, the project will not compile.

## Initial Setup

1. Fork the Passes repository on GitHub (see
   <https://github.com/Munich-Quantum-Software-Stack/passes>).

2. Clone your fork locally

   ```sh
   git clone https://forked-url/passes
   ```

3. Change into the project directory

   ```sh
   cd passes
   ```

4. Create a branch for local development

   ```sh
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

## Alternative: Visual Studio Code DevContainer (Strongly recommended)

**Dev Containers** (short for Development Containers) are an environment configuration that allows
developers to create a consistent, isolated environment for development, testing, and deployment.
They are typically used in conjunction with Visual Studio Code (**VS Code**) and **Docker** to
ensure that the development environment is the same across all team members, regardless of their
host machine or operating system.

- Open the repository in VS-code
- Use the Dev Container plugin to Open Folder in Container. This should pull and install LLVM/MLIR
  and the required dependencies.

The previous steps creates an isolated environment where you can tests this project. The project is
located:

```shell
cd /workspaces/passes
```

## Working on Source Code

Building the project requires a C++ compiler supporting _C11_ and a minimum CMake version of _3.19_.
The example devices and the tests require a C++ compiler supporting _C++17_ and _C++20_ dialects.

### Configure and Build

This collection of MLIR passes uses CMake as its build system. However, we provide an script
`build.sh` that first builds CudaQ library `cudaq-mlir-runtime`. Then, the script builds the MQSS
passes project. You can configure and build the project as follows:

```shell
bash build.sh --jobs 5 --debug --mlir-dir "dir-to-mlir" --clang-dir "dir-to-clang"
              --llvm-dir "dir-to-llvm" --build-tests --build-docs --build-tools
```

In the following, we describe each of the arguments accepted by `build.sh`.

- `--jobs` (**optional**) The number of jobs utilized to configure and compile the project. If not
  specified, a single job is used to compile the project.
- `--debug` (**optional**) Used to show debug information. If you want to build the project without
  debug information, do not include it.
- `--mlir-dir` Path of your MLIR installation. E.g., `/opt/llvm/lib/cmake/mlir/`
- `--clang-dir` Path of your Clang installation.
- `--llvm-dir` Path of your LLVM installation.
- `--build-tests` (**optional**) Include it if you want to build the tests of this project.
- `--build-docs` (**optional**) Include it if you want to build this documentation.
- `--build-tools` (**optional**) Include it if you want to build additional tools.

### Running Tests

We use the [GoogleTest](https://google.github.io/googletest/primer.html) framework for unit testing
each MLIR pass in this collection. All tests are contained in the `test` directory. You can
configure and build the project using CMake as follows:

```shell
bash build.sh --build-tests
```

The executables used to run the tests can be found at `build/tests/`. To verify all the available
tests, run:

```shell
ctest --test-dir build --output-on-failure
```

### Format for Comments

For the information to be displayed correctly in the documentation, it is essential that the
comments follow the format required by Doxygen. Below you find some tags, that are commonly used
within the documentation of a function:

- `@brief` For a brief, one-line description of the function. Should always be provided.
- `@details` For a longer, detailed description of the function.
- `@param` To explain the usage of a parameter. Should be provided for each parameter.
- `@return` To explain the return value. Should be provided if the function returns a value.

\note In the current setting, the long description is always prepended with the brief description.
So there is no need to repeat the brief description in the details.

## Working on the Documentation

The documentation is generated using [Doxygen](https://www.doxygen.nl/index.html), which is
seamlessly integrated into the CMake build system.

### Building the Documentation

The documentation can be built configuring the CMake as follows:

```shell
bash build.sh --build-docs
```

The generated webpage can be inspected by opening the file in `docs/html/index.html` in the CMake
build directory.

### Static Content

The generated webpage also contains four static sites, namely the main page, the support page, the
FAQ page, and this development guide. The respective markdown files that serve as the source for
those sites are contained in `docs/` where `index.md` contains the content of the main page.

### Dynamic Content

In order to include source files to be listed among the menu item `API Reference/Files`, these files
must be marked as documented. This is accomplished by adding a comment like the following to the top
of the file. Right now, this is done for all files in the include directory.

### Further Links

- For more details, see the official documentation of Doxygen that can be found here:
  [https://www.doxygen.nl/manual/docblocks.html](https://www.doxygen.nl/manual/docblocks.html).
- More tags and commands can be found in the list provided here:
  [https://www.doxygen.nl/manual/commands.html#cmd_intro](https://www.doxygen.nl/manual/commands.html#cmd_intro)
