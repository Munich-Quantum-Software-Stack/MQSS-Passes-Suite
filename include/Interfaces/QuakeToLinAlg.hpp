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
 * @brief Given a gate as a quake::OperatorInterface, this functions inlines the
 corresponding matrix in the given builder.
   @details This method inlines a matrix corresponding to gate into an MLIR
 module associated with the builder passed as parameter.
    @param[in] gate is the mlir gate to be inserted.
    @param[out] builder is an `OpBuilder` object associated with a MLIR module.
 It is used to insert new instructions to the corresponding MLIR module.
    @param[in] loc is the location of the new inserted instruction.
*/
// TODO
void insertGatesToMLIRModule(mlir::ModuleOp module, QuakeDAG dag,
                             OpBuilder &builder, func::FuncOp gpuFunction);

} // namespace mqss::interfaces
