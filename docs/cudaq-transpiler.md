# CudaQ Transpiler

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

__CudaQ__ offers also a transpiler infrastructure that is used by `nvq++` quantum compiler. Transpilation can also be handled using MLIR passes as follows:

First, extract the MLIR context and define an MLIR `PassManager` as follows:

```cpp
auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
mlir::MLIRContext &context = *contextPtr;
// creating pass manager
mlir::PassManager pm(&context);
```
Next, define a `BasisConversionPassOptions` object and pass to it a list of gates to be considered as the native gate set. The transpiler will use the defined native gate set to try to apply decomposition passes. For instance, the following native gate set `{"h",  "s", "t", "rx", "ry", "rz", "x", "y", "z",  "x(1)"}` is specified.

```cpp
cudaq::opt::BasisConversionPassOptions options;
options.basis = {"h",  "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"};
```

Next, add a `cudaq::opt::createBasisConversionPass` to the pass manager. Do not forget to pass  `options` as input parameter of `cudaq::opt::createDecompositionPass`, as follows:

```cpp
pm.addPass(createBasisConversionPass(options));
```

Finally, you can dump and visualize the effects of the decomposition pattern on your MLIR module as follows:

```cpp
// running the pass
if(mlir::failed(pm.run(mlirModule)))
  std::runtime_error("The pass failed...");
// if pass is applied succesfully ...
std::cout << "Circuit after pass:\n";
mlirModule->dump();
```

The following circuit will be used as a test case. Each following sections shows the transpiled circuit obtained for __IQM, AQT, PlanQ and WMI__ backends.

<div align="center">
  <img  alt="Input" src="Input.png" width=75%>
</div>

## AQT

The AQT backend supports the following gates: `"x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "x(1)", "z(1)", "swap"`. 
 
![AQT](AQTTranspilation.png)

## WMI

The WMI backend supports the following gates: `"rx", "ry", "rz", "h", "phased_rx", "phased_ry", "phased_rz", "x(1)", "z(1)"`. 

<div align="center">
  <img  alt="WMI" src="WMITranspilation.png" width=100%>
</div>

## PlanQ
The WMI backend supports the following gates: `"rx", "ry", "rz", "x(1)", "z(1)"`. 

TODO

## IQM
The IQM backend supports the following gates: `"phased_rx","z(1)"`. For the sake of clarity, just fragments of the transpiled circuit are shown.

### 1)
<div align="center">
  <img  alt="Fragment 1" src="Transpiled-IQM1.png" width=100%> 
</div>

### 2)
<div align="center">
  <img  alt="Fragment 2" src="Transpiled-IQM2.png" width=100%>
</div>

### ...

### n)
<div align="center">
  <img  alt="Fragment 3" src="Transpiled-IQMn.png" width=100%>
</div>
