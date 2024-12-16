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
---------------------------------------------------------------------------->

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/_static/mqss_logo_dark.svg" width="20%">
    <img src="./docs/_static/mqss_logo.svg" width="20%">
  </picture>
</div>

# Collection of MLIR Passes of the MQSS

<div align="center">
  <!-- <a href="https://munich-quantum-software-stack.github.io/QDMI/"> -->
  <img style="min-width: 200px !important; width: 30%;" src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNi4wIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjQgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZmlsbD0iI2ZmZmZmZiIgZD0iTTk2IDBDNDMgMCAwIDQzIDAgOTZMMCA0MTZjMCA1MyA0MyA5NiA5NiA5NmwyODggMCAzMiAwYzE3LjcgMCAzMi0xNC4zIDMyLTMycy0xNC4zLTMyLTMyLTMybDAtNjRjMTcuNyAwIDMyLTE0LjMgMzItMzJsMC0zMjBjMC0xNy43LTE0LjMtMzItMzItMzJMMzg0IDAgOTYgMHptMCAzODRsMjU2IDAgMCA2NEw5NiA0NDhjLTE3LjcgMC0zMi0xNC4zLTMyLTMyczE0LjMtMzIgMzItMzJ6bTMyLTI0MGMwLTguOCA3LjItMTYgMTYtMTZsMTkyIDBjOC44IDAgMTYgNy4yIDE2IDE2cy03LjIgMTYtMTYgMTZsLTE5MiAwYy04LjggMC0xNi03LjItMTYtMTZ6bTE2IDQ4bDE5MiAwYzguOCAwIDE2IDcuMiAxNiAxNnMtNy4yIDE2LTE2IDE2bC0xOTIgMGMtOC44IDAtMTYtNy4yLTE2LTE2czcuMi0xNiAxNi0xNnoiLz48L3N2Zz4=" alt="Documentation" />
  </a>
</div>

## FAQ

### What is MQSS?

**MQSS** stands for _Munich Quantum Software Stack_, which is a project of the _Munich Quantum
Valley (MQV)_ initiative and is jointly developed by the _Leibniz Supercomputing Centre (LRZ)_ and
the Chairs for _Design Automation (CDA)_, and for _Computer Architecture and Parallel Systems
(CAPS)_ at TUM. It provides a comprehensive compilation and runtime infrastructure for on-premise
and remote quantum devices, support for modern compilation and optimization techniques, and enables
both current and future high-level abstractions for quantum programming. This stack is designed to
be capable of deployment in a variety of scenarios via flexible configuration options, including
stand-alone scenarios for individual systems, cloud access to a variety of devices as well as tight
integration into HPC environments supporting quantum acceleration. Within the MQV, a concrete
instance of the MQSS is deployed at the LRZ for the MQV, serving as a single access point to all of
its quantum devices via multiple compatible access paths, including a web portal, command line
access via web credentials as well as the option for hybrid access with tight integration with LRZ's
HPC systems. It facilitates the connection between end-users and quantum computing platforms by its
integration within HPC infrastructures, such as those found at the LRZ.

### What is MLIR?
MLIR (Multi-Level Intermediate Representation) is a versatile compiler framework for developing domain-specific compilers and optimizing transformations. MLIR originated as part of the LLVM ecosystem and is particularly tailored for modern, complex computational workflows, including machine learning, AI, and heterogeneous hardware.
MLIR Supports multiple levels of abstraction within a single framework, allowing developers to work with high-level domain-specific operations down to hardware-specific operations. Users can define their dialects (custom operations and types) for specific problem domains while leveraging the shared infrastructure for optimization and code generation.
Additionally, MLIR promotes interoperability among different models of computation and supports optimization passes across various abstraction levels, including high-level and low-level operations.

<div align="center">
  <picture>
    <img src="./docs/_static/mlir.png" width="60%">
  </picture>
</div>

### What is an MLIR pass?

<div align="center">
  <picture>
    <img src="./docs/_static/mlir-pass.png" width="45%">
  </picture>
</div>


<div align="center">
  <picture>
    <img src="./docs/_static/mlir-passes.png" width="60%">
  </picture>
</div>

### Why to include MLIR into the MQSS?
<div align="center">
  <picture>
    <img src="./docs/_static/mlir-why.png" width="65%">
  </picture>
</div>

### What is the QUAKE MLIR Dialect?
<div align="center">
  <picture>
    <img src="./docs/_static/mlir-quake.png" width="40%">
  </picture>
</div>


### Where does MLIR passes fit into the MQSS?

<div align="center">
  <picture>
    <img src="./docs/_static/mlir-fit.png" width="60%">
  </picture>
</div>

### Where is the code?

The code is publicly available and hosted on GitHub at

### Under which license is this collection of passes released?

This collection of MLIR passes is released under the Apache License v2.0 with LLVM Exceptions. See
[LICENSE](https://github.com/Munich-Quantum-Software-Stack/QDMI/blob/develop/LICENSE) for more
information. Any contribution to the project is assumed to be under the same license.
