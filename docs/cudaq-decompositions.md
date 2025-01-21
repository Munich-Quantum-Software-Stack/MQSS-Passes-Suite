# CudaQ Decomposition Patterns

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

__CudaQ__ offers a set of decomposition passes used by the `nvq++` quantum compiler. The passes can be applied to a given quantum `kernel` as follows:

First, extract the MLIR context and define an MLIR `PassManager` as follows:

```cpp
auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
mlir::MLIRContext &context = *contextPtr;
// creating pass manager
mlir::PassManager pm(&context);
```

Next, define a `DecompositionPassOptions` object and pass to it a list with the decomposition passes that you desire to apply. For instance, in the following example the decomposition pattern `CXToCZ` is specified.

```cpp
cudaq::opt::DecompositionPassOptions options;
options.enabledPatterns = {"CXToCZ"};
```
\note The `enabledPatterns` in the  `cudaq::opt::DecompositionPassOptions` is a list. Thus, more than one decomposition pattern can be applied using the same pass manager.

Next, add a `cudaq::opt::createDecompositionPass` to the pass manager. Do not forget to pass  `options` as input parameter of `cudaq::opt::createDecompositionPass`, as follows:

```cpp
pm.addPass(cudaq::opt::createDecompositionPass(options));
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

In the following, the list of all decomposition patterns offered by __CudaQ__ are shown with a respective example.

## CCX to CCZ

The pattern `CCXToCCZ` replaces all the __two-controls X__ gates in a circuit by __two-controls Z__ gates.

<div align="center">
  <img  alt="Pass CCXToCCZ" src="CCXToCCZ.png" width=75%>
</div>

## CCZ to CX
The pattern `CCZToCX` replaces all the __two-controls Z__ gates in a circuit by __two-controls X__ gates.
![Pass CCZToCX](CCZToCX.png)  

## CH to CX
The pattern `CHToCX` replaces all the __controlled Hadamard__ gates in a circuit by __CNot__ gates.
![Pass CHToCX](CHToCX.png)

## CR1 to CX
The pattern `CR1ToCX` replaces all the __controlled R1__ gates in a circuit by __CNot__ gates.
![Pass CR1ToCX](CR1ToCX.png)

## CRx to CX
The pattern `CRxToCX` replaces all the __controlled RX__ gates in a circuit by __CNot__ gates.
![Pass CRxToCX](CRxToCX.png)

## CRy to CX
The pattern `CRyToCX` replaces all the __controlled RY__ gates in a circuit by __CNot__ gates.
![Pass CRyToCX](CRyToCX.png)

## CRz to CX
The pattern `CRzToCX` replaces all the __controlled RZ__ gates in a circuit by __CNot__ gates.
![Pass CRzToCX](CRzToCX.png)

## CX to CZ
The pattern `CXToCZ` replaces all the __CNot__ gates in a circuit by __controlled Z__ gates.

<div align="center">
  <img  alt="Pass CXToCZ" src="CXToCZ.png" width=75%>
</div>

## CZ to CX
The pattern `CZToCX` replaces all the __controlled Z__ gates in a circuit by __CNot__ gates.

<div align="center">
  <img  alt="Pass CZToCX" src="CZToCX.png" width=75%>
</div>

## Exp Pauli Decomposition
The pattern `ExpPauliDecomposition` applies __Pauli Decompositions__.
<div align="center">
  <img  alt="Pass ExpPauliDecomposition" src="ExpPauliDecomposition.png" width=75%>
</div>

## H to PhasedRx
The pattern `HToPhasedRx` replaces all the __Hadamard__ gates in a circuit by __phased RX__ gates.
<div align="center">
  <img  alt="Pass HToPhasedRx" src="HToPhasedRx.png" width=85%>
</div>

## R1 to PhasedRx
The pattern `R1ToPhasedRx` replaces all the __R1__ gates in a circuit by __phased RX__ gates.
![Pass R1ToPhasedRx](R1ToPhasedRx.png)

## R1 to Rz
The pattern `R1ToRz` replaces all the __R1__ gates in a circuit by __RZ__ gates.

<div align="center">
  <img  alt="Pass R1ToRz" src="R1ToRz.png" width=75%>
</div>

## Rx to PhasedRx
The pattern `RxToPhasedRx` replaces all the __RX__ gates in a circuit by __phased RX__ gates.
![Pass RxToPhasedRx](RxToPhasedRx.png) 

## Ry to PhasedRx
The pattern `RyToPhasedRx` replaces all the __RY__ gates in a circuit by __phased RX__ gates.
![Pass RyToPhasedRx](RyToPhasedRx.png)

## Rz to PhasedRx
The pattern `RzToPhasedRx` replaces all the __RZ__ gates in a circuit by __phased RX__ gates.
![Pass RzToPhasedRx](RzToPhasedRx.png)

## S to PhasedRx
The pattern `SToPhasedRx` replaces all the __S__ gates in a circuit by __phased RX__ gates.
![Pass SToPhasedRx](SToPhasedRx.png)

## S to R1
The pattern `SToR1` replaces all the __S__ gates in a circuit by __R1__ gates.

<div align="center">
  <img  alt="Pass SToR1" src="SToR1.png" width=75%>
</div>

## Swap to CX
The pattern `SwapToCX` replaces all the __swap__ gates in a circuit by __CNot__ gates.
<div align="center">
  <img  alt="Pass SwapToCX" src="SwapToCX.png" width=75%>
</div>

## T to PhasedRx
The pattern `TToPhasedRx` replaces all the __T__ gates in a circuit by __phased RX__ gates.
![Pass TToPhasedRx](TToPhasedRx.png)

## T to R1
The pattern `TToR1` replaces all the __T__ gates in a circuit by __R1__ gates.
<div align="center">
  <img  alt="Pass TToR1" src="TToR1.png" width=75%>
</div>

## U3 to Rotations
The pattern `U3ToRotations` replaces all the __U3__ gates in a circuit by __rotation__ gates.
![Pass U3ToRotations](U3ToRotations.png)

## X to PhasedRx
The pattern `XToPhasedRx` replaces all the __X__ gates in a circuit by __phased RX__ gates.
<div align="center">
  <img  alt="Pass XToPhasedRx" src="XToPhasedRx.png" width=75%>
</div>

## Y to PhasedRx
The pattern `YToPhasedRx` replaces all the __Y__ gates in a circuit by __phased RX__ gates.
<div align="center">
  <img  alt="Pass YToPhasedRx" src="YToPhasedRx.png" width=75%>
</div>

## Z to PhasedRx
The pattern `ZToPhasedRx` replaces all the __Z__ gates in a circuit by __phased RX__ gates.
![Pass ZToPhasedRx](ZToPhasedRx.png)
