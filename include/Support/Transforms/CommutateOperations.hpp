/* This code and any associated documentation is provided "as is"

Copyright 2025 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

TODO: URL LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------
  author Martin Letras
  date   February 2025
  version 1.0
  brief
    TODO

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

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
// Finds the pattern composed of T2, T1 and commute them to T1, T2
template <typename T1, typename T2>
void commuteOperation(mlir::Operation *currentOp, int nCtrlsOp1, int nTgtsOp1,
                      int nCtrlsOp2, int nTgtsOp2) {
  auto currentGate = dyn_cast_or_null<T2>(*currentOp);
  if (!currentGate)
    return;
  // check that the current gate is compliant with the number of controls and
  // targets
  if (currentGate.getControls().size() != nCtrlsOp2 ||
      currentGate.getTargets().size() != nTgtsOp2)
    return;
  // get the previous operation to check the swap pattern
  auto prevOp =
      getPreviousOperationOnTarget(currentGate, currentGate.getTargets()[0]);
  if (!prevOp)
    return;
  auto previousGate = dyn_cast_or_null<T1>(prevOp);
  if (!previousGate)
    return;
  // check that the previous gate is compliant with the number of controls and
  // targets
  if (previousGate.getControls().size() != nCtrlsOp1 ||
      previousGate.getTargets().size() != nTgtsOp1)
    return; // check both targets are the same
  int targetPrev = extractIndexFromQuakeExtractRefOp(
      previousGate.getTargets()[0].getDefiningOp());
  int targetCurr = extractIndexFromQuakeExtractRefOp(
      currentGate.getTargets()[0].getDefiningOp());
  if (targetPrev != targetCurr)
    return;
#ifdef DEBUG
  llvm::outs() << "Current Operation: ";
  currentGate->print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs() << "Previous Operation: ";
  previousGate->print(llvm::outs());
  llvm::outs() << "\n";
#endif
  // At this point, I should de able to do the commutation
  // Swap the two operations by cloning them in reverse order.
  mlir::IRRewriter rewriter(currentGate->getContext());
  rewriter.setInsertionPointAfter(currentGate);
  auto newGate = rewriter.create<T1>(
      previousGate.getLoc(), previousGate.isAdj(), previousGate.getParameters(),
      previousGate.getControls(), previousGate.getTargets());
  rewriter.setInsertionPoint(newGate);
  rewriter.create<T2>(currentGate.getLoc(), currentGate.isAdj(),
                      currentGate.getParameters(), currentGate.getControls(),
                      currentGate.getTargets());
  // Erase the original operations
  rewriter.eraseOp(currentGate);
  rewriter.eraseOp(previousGate);
}
} // namespace mqss::support::transforms
