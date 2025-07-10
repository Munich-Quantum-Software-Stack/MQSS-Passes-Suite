/* This code and any associated documentation is provided "as is"

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
******************************************************************************
  author Martin Letras
  date   July 2025
  version 1.0
******************************************************************************/
/** @file
 * @brief
 * @details This header defines a set of functions utilized to convert Quake
 * circuits to Arith + LinAlg
 *
 * @par
 * This header file is used by the QASM3ToQuakePass to perform the conversion of
 * QASM programs to MLIR/Quake modules.
 */

#pragma once

#include "Support/DAG/Quake-DAG.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mqss::interfaces {

/**
 * @brief Given a module and a quake quantum kernel. This method perform the
 transformation from Quake to LinAlg + Arith.
   @details This method converts any given Quake quantum kernel to a sequence of
 complex vectors/matrices multiplications corresponding to the input quantum
 kernel. This representation might be useful, later as input of IREE compiler
 which is able to generate GPU code for many vendors.
    @param[out] module is the mlir module where my input quantum kernel is. In
 module the converted function will be inserted.
    @param[in] quakeFunction is the quantum kernel to be converted.
    @param[in] builder is an `OpBuilder` object associated with a MLIR module.
 It is used to insert new instructions to the corresponding MLIR module.
    @param[in] tensorType is the datatype associated to the state vector
    @param[in] matrixType is the datatype associated to the gate matrices
    @param[in] numberOfQubits is the number of qubits utilized by the quantum
 kernel
*/
mlir::Value convertQuakeToLinAlg(mlir::ModuleOp module,
                                 mlir::func::FuncOp quakeFunction,
                                 OpBuilder &builder, func::FuncOp gpuFunction,
                                 mlir::RankedTensorType tensorType,
                                 mlir::RankedTensorType matrixType,
                                 int numberOfQubits);

} // namespace mqss::interfaces
