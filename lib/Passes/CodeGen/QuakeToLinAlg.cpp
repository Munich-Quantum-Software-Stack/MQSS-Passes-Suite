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
-------------------------------------------------------------------------
  author Martin Letras
  date   July 2025
  version 1.0
  brief
    MLIR pass that converts an input quantum kernel in Quake to LinAlg + Arith

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Interfaces/QuakeToLinAlg.hpp"

#include "Passes/CodeGen.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <iomanip>
#include <regex>
#include <unordered_map>

using namespace mlir;
using namespace mqss::support::quakeDialect;
using namespace mqss::interfaces;

namespace {

bool hasFunc(mlir::ModuleOp module, llvm::StringRef name) {
  return static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>(name));
}

class QuakeToLinAlg
    : public PassWrapper<QuakeToLinAlg, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuakeToLinAlg)

  QuakeToLinAlg() {}

  llvm::StringRef getArgument() const override {
    return "convert-quake-to-linalg";
  }
  llvm::StringRef getDescription() const override {
    return "Convert Quake to linalg Operations";
  };

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    auto context = &getContext();
    context->getOrLoadDialect<mlir::tensor::TensorDialect>();
    // Iterate through all functions
    std::vector<func::FuncOp> quakeKernels;
    for (auto func : module.getOps<func::FuncOp>()) {
      // Check if the function has the "cudaq-kernel" attribute
      if (!func->hasAttr("cudaq-kernel"))
        continue;
      llvm::StringRef functionName = func.getName();
      int numQubits = getNumberOfQubits(func);
      mlir::Location loc = builder.getUnknownLoc();
      // Set insertion point inside the module
      builder.setInsertionPointToStart(module.getBody());
      // Create type of states of size  [2^numberQubits, 1]
      auto elementType = mlir::ComplexType::get(builder.getF64Type());
      mlir::RankedTensorType tensorType =
          mlir::RankedTensorType::get({std::pow(2, numQubits)}, elementType);
      // Create the type of the matrices of each operation
      // The matrices has to be [2^numQubits, 2^numQubits]
      auto shape = llvm::SmallVector<int64_t>{std::pow(2, numQubits),
                                              std::pow(2, numQubits)};
      auto matrixType = mlir::RankedTensorType::get(shape, elementType);
      // the gpu function should be able to return a tensor of [2^numberQubits,
      // 1]
      auto funcType = builder.getFunctionType({}, {tensorType});
      auto gpuFunc = builder.create<mlir::func::FuncOp>(
          loc, "mqss_gpu" + functionName.str(), funcType);
      // Agregar bloque de entrada vacío
      mlir::Block *entryBlock = gpuFunc.addEntryBlock();
      // Insert instructions
      builder.setInsertionPointToStart(entryBlock);
      mlir::Value result = convertQuakeToLinAlg(
          module, func, builder, gpuFunc, tensorType, matrixType, numQubits);
      builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), result);
      quakeKernels.push_back(func);
    }
    // deleting quake kernels after they are converted
    for (auto f : quakeKernels)
      f.erase();
  }

private:
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQuakeToLinAlgPass() {
  return std::make_unique<QuakeToLinAlg>();
}
