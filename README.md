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
---------------------------------------------------------------------------->

<div align="center">
  <a href="https://munich-quantum-software-stack.github.io/MQSS-Passes-Documentation/mlir/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/_static/mqss_logo_dark.svg" width="20%">
    <img src="./docs/_static/mqss_logo.svg" width="20%">
  </picture>
</div>

# Collection of MLIR Passes of the MQSS

<!-- [DOXYGEN MAIN] -->

This repository holds a collection of MLIR passes that operate on Quantum programs to optimize,
transform, and lower to target devices. The presented passes are integrated into the Munich Quantum
Software Stack (MQSS) infrastructure. In particular, this collection of passes is used in the
Quantum Resource Manager (QRM) for optimizing, transforming, and lowering quantum programs to
quantum devices. The passes stored in this collection can be classified as target-agnostic and
target-specific. Target agnostic passes can be applied to any quantum circuit and do not require
information on the selected quantum target device. In contrast, target-specific passes tightly
depend on the selected quantum device. For instance, transpilation passes that convert a quantum
circuit defined using arbitrary gates to a quantum circuit compliant with the native gate set of the
selected quantum device.

<!-- [DOXYGEN MAIN] -->
<div align="center">
  <img style="min-width: 200px !important; width: 30%;" src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNi4wIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjQgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZmlsbD0iI2ZmZmZmZiIgZD0iTTk2IDBDNDMgMCAwIDQzIDAgOTZMMCA0MTZjMCA1MyA0MyA5NiA5NiA5NmwyODggMCAzMiAwYzE3LjcgMCAzMi0xNC4zIDMyLTMycy0xNC4zLTMyLTMyLTMybDAtNjRjMTcuNyAwIDMyLTE0LjMgMzItMzJsMC0zMjBjMC0xNy43LTE0LjMtMzItMzItMzJMMzg0IDAgOTYgMHptMCAzODRsMjU2IDAgMCA2NEw5NiA0NDhjLTE3LjcgMC0zMi0xNC4zLTMyLTMyczE0LjMtMzIgMzItMzJ6bTMyLTI0MGMwLTguOCA3LjItMTYgMTYtMTZsMTkyIDBjOC44IDAgMTYgNy4yIDE2IDE2cy03LjIgMTYtMTYgMTZsLTE5MiAwYy04LjggMC0xNi03LjItMTYtMTZ6bTE2IDQ4bDE5MiAwYzguOCAwIDE2IDcuMiAxNiAxNnMtNy4yIDE2LTE2IDE2bC0xOTIgMGMtOC44IDAtMTYtNy4yLTE2LTE2czcuMi0xNiAxNi0xNnoiLz48L3N2Zz4=" alt="Documentation" />
  </a>
</div>

## FAQ

<!-- [DOXYGEN FAQ] -->

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

MLIR (Multi-Level Intermediate Representation) is a versatile compiler framework for developing
domain-specific compilers and optimizing transformations. MLIR originated as part of the LLVM
ecosystem and is particularly tailored for modern, complex computational workflows, including
machine learning, AI, and heterogeneous hardware.

MLIR supports multiple levels of abstraction within a single framework, allowing developers to work
with high-level domain-specific operations down to hardware-specific operations. Users can define
their **dialects** (custom operations and types) for specific problem domains while leveraging the
shared infrastructure for optimization and code generation.

Additionally, MLIR promotes interoperability among different models of computation and supports
**optimization passes** across various abstraction levels, including high-level and low-level
operations.

<div align="center">
    <img src="./docs/_static/mlir.png" width="60%">
</div>

For more information on [MLIR](https://github.com/llvm/llvm-project.git).

### What is an MLIR Dialect?

An MLIR dialect is a modular and extensible namespace within the MLIR framework that defines a set
of **operations**, **types** and **attributes** specific to a domain, language, or computation
model. Dialects enable MLIR to be a highly flexible intermediate representation (IR).

For instance, **Quake** is an MLIR dialect designed for quantum computing. It serves as part of
NVIDIA's CUDAQ framework, which facilitates the development, optimization, and deployment of
quantum-classical hybrid programs. Quake represents quantum programs within MLIR, providing a
high-level abstraction for quantum operations and allowing developers to leverage the MLIR
infrastructure for optimization and compilation.

<div align="center">
    <img src="./docs/_static/mlir-quake.png" width="60%">
</div>

For more information on [QUAKE MLIR Dialect](https://github.com/NVIDIA/cuda-quantum.git).

### What is an MLIR pass?

An MLIR pass is a transformation or analysis applied to an MLIR intermediate representation (IR) to
**modify**, **optimize**, or **gather information**. Passes are a central concept in compiler
frameworks, including MLIR, enabling modular, reusable, and extensible transformations of code at
various levels of abstraction.

<div align="center">
    <img src="./docs/_static/mlir-pass.png" width="45%">
</div>

For instance, in the figure above, an MLIR optimization pass is applied to the input circuit, which
contains two consecutive Hadamard gates on qubit 0. Accordingly, in the output-optimized circuit
shown on the right, those two consecutive Hadamards are removed because they are equivalent to an
identity operation.

MLIR has two categories of passes: **transformation** passes, and **analysis** passes. The pass
presented above is a transformation pass. Moreover, passes can be applied in sequences defined as
**pass pipelines**.

<div align="center">
    <img src="./docs/_static/mlir-passes.png" width="60%">
</div>

In the figure shown above, three pass pipelines are defined. In purple, a synthesis to QUAKE
pipeline synthesizes QUAKE MLIR code from a given input C++ program. In green, an optimization
pipeline that applies a series of transformations passes on MLIR modules. Finally, in orange, a pass
pipeline that lowers QUAKE MLIR modules to the Quantum Intermediate Representation (QIR).

### Why to include MLIR into the MQSS?

One fundamental feature of MLIR is its ability to model different levels of abstraction related to a
domain-specific language. In contrast, Quantum representations such as the QIR or QASM are low-level
representations. Performing transformations on a low-level abstraction might not be a good choice.
Low-level representations are a list of quantum gates that operate on qubits. In the example below,
the QIR program on the right is equivalent to the quantum circuit on the left.

<div align="center">
    <img src="./docs/_static/mlir-why.png" width="65%">
</div>

However, dataflow dependencies are lost in low-level representations, such as QIR. In contrast, MLIR
(QUAKE) holds the dataflow dependencies natively, and no modifications to the compilation
infrastructure are required. Thus, transformation passes such as decompositions or replacements can
be easily implemented. Moreover, other dialects can be integrated with QUAKE to re-utilize the
existing MLIR infrastructure.

### Where do MLIR passes fit into the MQSS?

The collection of MLIR passes stored in this repository is part of the Munich Quantum Software Stack
(MQSS). The passes are utilized inside the Quantum Resource Manager
([QRM](https://github.com/Munich-Quantum-Software-Stack/QRM)).

<div align="center">
    <img src="./docs/_static/mlir-fit.png" width="60%">
</div>

For example, a hybrid quantum application can be defined in a high-level programming language such
as C++ using the CUDAQ library. Those code fragments in a hybrid quantum application defined as
quantum kernels will be submitted to the QRM.

By specifying the target as MQSS at compilation time, the generated binary will orchestrate the
classical and quantum resources. Every time a quantum kernel has to be executed, the compiled binary
submits the MLIR code of the correspondent quantum kernel to the MQSS stack. Thus, on the MQSS side,
the MLIR module is processed by the QRM, which perform **agnostic passes** (optimization passes) and
**target-specific passes** (transpilation passes and lowering passes). The lowered code is sent to
the Quantum device, and the results are collected and sent back to the hybrid application.

### Where is the code?

The code is publicly available and hosted on GitHub:
https://github.com/Munich-Quantum-Software-Stack/passes.

### Under which license is this collection of passes released?

This collection of MLIR passes is released under the Apache License v2.0 with LLVM Exceptions. See
[LICENSE](https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE) for more
information. Any contribution to the project is assumed to be under the same license.

<!-- [DOXYGEN FAQ] -->
