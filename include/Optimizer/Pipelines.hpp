#pragma once

#include "mlir/Conversion/MathToFuncs/MathToFuncs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mqss::opt {

void O1(mlir::PassManager &pm);
void O2(mlir::PassManager &pm);
void O3(mlir::PassManager &pm);

} // namespace mqss::opt
