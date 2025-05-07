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
*******************************************************************************
  author Martin Letras
  date   February 2025
  version 1.0
*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/
/** @file
  @brief
  @details Definition of the switch operation. This function is only valid for
two qubit gates. Given a pattern using the template, the function will find the
pattern and perform a switch operation.
  @par
  This header must be included to use the switch operations to manipulate MLIR
  modules.
*/

#pragma once

#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mqss::support::quakeDialect;

namespace mqss::support::transforms {

/**
 * @brief Detects and rewrites a pattern of four quantum operations.
 *
 * @details  Finds the pattern composed of T1, T2 and switches them
 * and assigns the types T3, and T4. Targets and controls should be the same on
 * both. This only works at the moment for single qubit gates
 * @tparam T1 Type of the first operation in the pattern.
 * @tparam T2 Type of the second operation in the pattern.
 * @tparam T3 Type of the third operation in the pattern.
 * @tparam T4 Type of the fourth operation in the pattern.
 * @param currentOp The MLIR operation from which the pattern search begins.
 */
template <typename T1, typename T2, typename T3, typename T4>
void patternSwitch(mlir::Operation *currentOp) {
  auto currentGate = dyn_cast_or_null<T2>(*currentOp);
  if (!currentGate)
    return;
  // check single qubit T2 operation
  if (currentGate.getControls().size() != 0 ||
      currentGate.getTargets().size() != 1)
    return;
  // get previous
  auto prevOp =
      getPreviousOperationOnTarget(currentGate, currentGate.getTargets()[0]);
  if (!prevOp)
    return;
  auto prevGate = dyn_cast<T1>(prevOp);
  // check single qubit operation
  if (prevGate.getControls().size() != 0 || prevGate.getTargets().size() != 1)
    return;
// I found the pattern, then I remove it from the circuit
#ifdef DEBUG
  llvm::outs() << "Current Operation: ";
  currentGate->print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs() << "Previous Operation: ";
  prevGate->print(llvm::outs());
  llvm::outs() << "\n";
#endif
  mlir::IRRewriter rewriter(currentGate->getContext());
  rewriter.setInsertionPointAfter(currentGate);
  auto newGate = rewriter.create<T3>(
      currentGate.getLoc(), currentGate.isAdj(), currentGate.getParameters(),
      currentGate.getControls(), currentGate.getTargets());
  rewriter.setInsertionPointAfter(newGate);
  rewriter.create<T4>(prevGate.getLoc(), prevGate.isAdj(),
                      prevGate.getParameters(), prevGate.getControls(),
                      prevGate.getTargets());
  rewriter.eraseOp(prevGate);
  rewriter.eraseOp(currentGate);
}
} // namespace mqss::support::transforms
