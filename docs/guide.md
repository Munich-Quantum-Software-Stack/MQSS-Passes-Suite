# Development Guide

<!-- IMPORTANT: Keep the line above as the first line. -->
<!----------------------------------------------------------------------------
Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

TODO: LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------- -->

<!-- This file is a static page and included in the CMakeLists.txt file. -->

Ready to contribute to the Collection of MLIR Passes of the MQSS? This guide will help you get started.

## Initial Setup

1. Fork the [Passes](TODO) repository on GitHub (see <https://TODO>).

2. Clone your fork locally

   ```sh
   git clone TODO
   ```

3. Change into the project directory

   ```sh
   cd mlir-passes TODO
   ```

4. Create a branch for local development

   ```sh
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

## Working on Source Code

Building the project requires a C compiler supporting _C11_ and a minimum CMake version of _3.19_.
The example devices and the tests require a C++ compiler supporting _C++17_ and _C++20_.

### Configure and Build

This collection of MLIR passes uses CMake as its build system. Building a project using CMake as follows:

First, the project needs to be _configured_ by calling

```shell
mkdir build
cd buid
cmake ..
```

After the configuration, the project can be _built_ by calling

```shell
make 
```

### Running Tests

We use the [GoogleTest](https://google.github.io/googletest/primer.html) framework for unit testing each MLIR in this collection. All tests are contained in the `test` directory. You can configure and build the project using CMake as follows:

```shell
cd buid
cmake .. -DBUILD_MLIR_PASSES_TESTS=ON
make
```

The executable used to run the tests can be found at `build/tests/testMQSSPasses`.


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
cd buid
cmake .. -DBUILD_MLIR_PASSES_DOCS=ON
make
```

The generated webpage can be inspected by opening the file in `docs/html/index.html` in the CMake build directory.

### Static Content

The generated webpage also contains four static sites, namely the main page, the support page, the FAQ page, and this development guide. The respective markdown files that serve as the source for those sites are contained in `docs/` where `index.md` contains the content of the main page.

### Dynamic Content

In order to include source files to be listed among the menu item `API Reference/Files`, these files must be marked as documented. This is accomplished by adding a comment like the following to the top of the file. Right now, this is done for all files in the include directory.

### Further Links

- For more details, see the official documentation of Doxygen that can be found here:
  [https://www.doxygen.nl/manual/docblocks.html](https://www.doxygen.nl/manual/docblocks.html).
- More tags and commands can be found in the list provided here:
  [https://www.doxygen.nl/manual/commands.html#cmd_intro](https://www.doxygen.nl/manual/commands.html#cmd_intro)
