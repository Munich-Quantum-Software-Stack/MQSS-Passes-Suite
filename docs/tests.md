# Testing Custom Passes

<!-- IMPORTANT: Keep the line above as the first line. -->

<!----------------------------------------------------------------------------
Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/Munich-Quantum-Software-Stack/QDMI/blob/develop/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------- -->

<!-- This file is a static page and included in the ./CMakeLists.txt file. -->

This page contains example implementations of devices and other components of the software stack
that use QDMI. All examples distributed with QDMI are contained in the `examples/` directory in the
repository.

\tableofcontents

## Implementing a Device {#device}

Below you find mock implementations of two QDMI devices: One is implemented in C++ and the other one
in C.

\note Keep in mind, that even though the interface is defined in C, the device can be implemented in
C++ or any other language that supports the C ABI.

### Basic String Properties {#device-string}

Both implementations use an auxiliary macro to add the string properties to the device. For an
explanation of the macro, see the next section.

### Auxiliary Macros {#device-macros}

The following macro is used to add string properties to the device. The macro is used, e.g., in the
implementation of the dev function.


The usage of the two latter macros is demonstrated in the following sections.

### Integer or Enumeration Properties {#device-int-enumeration}

The following two examples demonstrate how to return integer or enumeration properties of the
device.

### List Properties {#device-list}

Some properties are returned as a list of various data types. The following example shows how to


## Complex Properties {#device-complex}
