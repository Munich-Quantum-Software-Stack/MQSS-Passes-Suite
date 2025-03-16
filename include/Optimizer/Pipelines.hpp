#pragma once

#include "mlir/Conversion/MathToFuncs/MathToFuncs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mqss::opt {

void o1(mlir::OpPassManager &pm);
void o2(mlir::OpPassManager &pm);
void o3(mlir::OpPassManager &pm);

} // namespace mqss::opt
