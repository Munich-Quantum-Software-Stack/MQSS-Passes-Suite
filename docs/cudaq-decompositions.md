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

**CudaQ** offers a set of decomposition passes used by the `nvq++` quantum compiler. The passes can
be applied to a given quantum `kernel` as follows:

First, extract the MLIR context and define an MLIR `PassManager` as follows:

```cpp
auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
mlir::MLIRContext &context = *contextPtr;
// creating pass manager
mlir::PassManager pm(&context);
```

Next, define a `DecompositionPassOptions` object and pass to it a list with the decomposition passes
that you desire to apply. For instance, in the following example the decomposition pattern `CXToCZ`
is specified.

```cpp
cudaq::opt::DecompositionPassOptions options;
options.enabledPatterns = {"CXToCZ"};
```

\note The `enabledPatterns` in the `cudaq::opt::DecompositionPassOptions` is a list. Thus, more than
one decomposition pattern can be applied using the same pass manager.

Next, add a `cudaq::opt::createDecompositionPass` to the pass manager. Do not forget to pass
`options` as input parameter of `cudaq::opt::createDecompositionPass`, as follows:

```cpp
pm.addPass(cudaq::opt::createDecompositionPass(options));
```

Finally, you can dump and visualize the effects of the decomposition pattern on your MLIR module as
follows:

```cpp
// running the pass
if(mlir::failed(pm.run(mlirModule)))
  std::runtime_error("The pass failed...");
// if pass is applied successfully ...
std::cout << "Circuit after pass:\n";
mlirModule->dump();
```

In the following, the list of all decomposition patterns offered by **CudaQ** are shown with a
respective example.

## CCX to CCZ

The pattern `CCXToCCZ` replaces all the **two-controls X** gates in a circuit by **two-controls Z**
gates.

<div align="center">
  <img  alt="Pass CCXToCCZ" src="CCXToCCZ.png" width=60%>
</div>

## CCZ to CX

The pattern `CCZToCX` replaces all the **two-controls Z** gates in a circuit by **two-controls X**
gates. ![Pass CCZToCX](CCZToCX.png)

## CH to CX

The pattern `CHToCX` replaces all the **controlled Hadamard** gates in a circuit by **CNot** gates.
![Pass CHToCX](CHToCX.png)

## CR1 to CX

The pattern `CR1ToCX` replaces all the **controlled R1** gates in a circuit by **CNot** gates.
![Pass CR1ToCX](CR1ToCX.png)

## CRx to CX

The pattern `CRxToCX` replaces all the **controlled RX** gates in a circuit by **CNot** gates.
![Pass CRxToCX](CRxToCX.png)

## CRy to CX

The pattern `CRyToCX` replaces all the **controlled RY** gates in a circuit by **CNot** gates.
![Pass CRyToCX](CRyToCX.png)

## CRz to CX

The pattern `CRzToCX` replaces all the **controlled RZ** gates in a circuit by **CNot** gates.
![Pass CRzToCX](CRzToCX.png)

## CX to CZ

The pattern `CXToCZ` replaces all the **CNot** gates in a circuit by **controlled Z** gates.

<div align="center">
  <img  alt="Pass CXToCZ" src="CXToCZ.png" width=75%>
</div>

## CZ to CX

The pattern `CZToCX` replaces all the **controlled Z** gates in a circuit by **CNot** gates.

<div align="center">
  <img  alt="Pass CZToCX" src="CZToCX.png" width=75%>
</div>

## Exp Pauli Decomposition

The pattern `ExpPauliDecomposition` applies **Pauli Decompositions**.

<div align="center">
  <img  alt="Pass ExpPauliDecomposition" src="ExpPauliDecomposition.png" width=65%>
</div>

## H to PhasedRx

The pattern `HToPhasedRx` replaces all the **Hadamard** gates in a circuit by **phased RX** gates.

<div align="center">
  <img  alt="Pass HToPhasedRx" src="HToPhasedRx.png" width=85%>
</div>

## R1 to PhasedRx

The pattern `R1ToPhasedRx` replaces all the **R1** gates in a circuit by **phased RX** gates.
![Pass R1ToPhasedRx](R1ToPhasedRx.png)

## R1 to Rz

The pattern `R1ToRz` replaces all the **R1** gates in a circuit by **RZ** gates.

<div align="center">
  <img  alt="Pass R1ToRz" src="R1ToRz.png" width=75%>
</div>

## Rx to PhasedRx

The pattern `RxToPhasedRx` replaces all the **RX** gates in a circuit by **phased RX** gates.

<div align="center">
  <img  alt="Pass RxToPhasedRx" src="RxToPhasedRx.png" width=90%>
</div>

## Ry to PhasedRx

The pattern `RyToPhasedRx` replaces all the **RY** gates in a circuit by **phased RX** gates.
![Pass RyToPhasedRx](RyToPhasedRx.png)

## Rz to PhasedRx

The pattern `RzToPhasedRx` replaces all the **RZ** gates in a circuit by **phased RX** gates.
![Pass RzToPhasedRx](RzToPhasedRx.png)

## S to PhasedRx

The pattern `SToPhasedRx` replaces all the **S** gates in a circuit by **phased RX** gates.
![Pass SToPhasedRx](SToPhasedRx.png)

## S to R1

The pattern `SToR1` replaces all the **S** gates in a circuit by **R1** gates.

<div align="center">
  <img  alt="Pass SToR1" src="SToR1.png" width=75%>
</div>

## Swap to CX

The pattern `SwapToCX` replaces all the **swap** gates in a circuit by **CNot** gates.

<div align="center">
  <img  alt="Pass SwapToCX" src="SwapToCX.png" width=75%>
</div>

## T to PhasedRx

The pattern `TToPhasedRx` replaces all the **T** gates in a circuit by **phased RX** gates.
![Pass TToPhasedRx](TToPhasedRx.png)

## T to R1

The pattern `TToR1` replaces all the **T** gates in a circuit by **R1** gates.

<div align="center">
  <img  alt="Pass TToR1" src="TToR1.png" width=75%>
</div>

## U3 to Rotations

The pattern `U3ToRotations` replaces all the **U3** gates in a circuit by **rotation** gates.
![Pass U3ToRotations](U3ToRotations.png)

## X to PhasedRx

The pattern `XToPhasedRx` replaces all the **X** gates in a circuit by **phased RX** gates.

<div align="center">
  <img  alt="Pass XToPhasedRx" src="XToPhasedRx.png" width=75%>
</div>

## Y to PhasedRx

The pattern `YToPhasedRx` replaces all the **Y** gates in a circuit by **phased RX** gates.

<div align="center">
  <img  alt="Pass YToPhasedRx" src="YToPhasedRx.png" width=75%>
</div>

## Z to PhasedRx

The pattern `ZToPhasedRx` replaces all the **Z** gates in a circuit by **phased RX** gates.
![Pass ZToPhasedRx](ZToPhasedRx.png)
