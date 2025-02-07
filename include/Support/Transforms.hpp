#ifndef SUPPORT_TRANSFORMS_H
#define SUPPORT_TRANSFORMS_H

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Support/Casting.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace mqss::support::transforms{

  // Finds the pattern composed of T1, T2 and switches them
  // and assigns the types T3, and T4
  // Targets and controls should be the same on boths
  // this only works at the moment for single qubit gates
  template <typename T1, typename T2, typename T3, typename T4>
  void patternSwitch(mlir::Operation *currentOp);

  // Finds the pattern composed of T2, T1 and remove them
  // and T1 and T2 share the same control
  // Targets and controls should be the same on boths
  template <typename T1, typename T2>
  void patternCancellation(mlir::Operation *currentOp, int nCtrlsOp1,
                           int nTgtsOp1, int nCtrlsOp2, int nTgtsOp2);

  // Finds the pattern composed of T2, T1 and commute them to T1, T2
  template <typename T1, typename T2>
  void commuteOperation(mlir::Operation *currentOp, int nCtrlsOp1, int nTgtsOp1,
                        int nCtrlsOp2, int nTgtsOp2);
}
#endif //SUPPORT_TRANSFORMS_H
