#ifndef SUPPORT_QUAKE_H
#define SUPPORT_QUAKE_H

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "llvm/Support/Casting.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace mqss::support::quakeDialect{
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
}

namespace supportQuake = mqss::support::quakeDialect;
#endif //SUPPORT_QUAKE_H
