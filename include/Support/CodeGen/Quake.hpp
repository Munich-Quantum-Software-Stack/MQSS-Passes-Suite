/* This code and any associated documentation is provided "as is"

Copyright 2025 Munich Quantum Software Stack Project

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
-------------------------------------------------------------------------
  author Martin Letras
  date   May 2025
  version 1.0
  brief
  This header defines a set of functions that are useful to get information from
  MLIR modules. E.g., extract the number of qubits, the classical registers.
  Moreover, it also allows to define variables as MLIR constructs.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mqss::support::quakeDialect {
// Given a OpBuilder and a double value, it inserts a double in the mlir
// module pointer by the OpBuilder and returns the inserted Value
Value createFloatValue(OpBuilder &builder, Location loc, double value);
// TODO: return -1 is not good idea
// Given an argument value as Operation, it extracts a double, it the operation
// is not double, returns -1.0 when fail
double extractDoubleArgumentValue(Operation *op);
// TODO: return -1 is not good idea
// Given an ExtractRefOp, it extracts the integer of the index pointing that
// reference (qubit index), returns -1 when fail
int64_t extractIndexFromQuakeExtractRefOp(Operation *op);
// function to get the number of qubits in a given quantum kernel
int getNumberOfQubits(func::FuncOp circuit);
// Function to get the number of classical bits allocated in a given
// quantum kernel, it also stores information of the qiubit position
int getNumberOfClassicalBits(func::FuncOp circuit,
                             std::map<int, int> &measurements);
// Function to get the number of classical bits allocated in
// a given quantum kernel
int getNumberOfClassicalBits(func::FuncOp circuit);
// Function that get the indices of the Value objectes in array
std::vector<int> getIndicesOfValueRange(mlir::ValueRange array);
// At the moment, it is assumed that the parameters are of type Double
std::vector<double> getParametersValues(mlir::ValueRange array);
// Get the previous operation on a given TargeQubit, starting from
// currentOp
mlir::Operation *getPreviousOperationOnTarget(mlir::Operation *currentOp,
                                              mlir::Value targetQubit);
// Get the next operation on a given TargeQubit, starting from
// currentOp
mlir::Operation *getNextOperationOnTarget(mlir::Operation *currentOp,
                                          mlir::Value targetQubit);
} // namespace mqss::support::quakeDialect

namespace supportQuake = mqss::support::quakeDialect;
