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
  date   February 2025
  version 1.0
  brief
    MLIR/Quake pass that converts an input QASM circuit to Quake.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

// #include "Interfaces/Constants.hpp"
// #include "Interfaces/QASMToQuake.hpp"
#include "Passes/CodeGen.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "ir/parsers/qasm3_parser/Parser.hpp"
#include "ir/parsers/qasm3_parser/Statement.hpp"
#include "ir/parsers/qasm3_parser/Types.hpp"
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
// using namespace mqss::interfaces;

namespace {

bool hasFunc(mlir::ModuleOp module, llvm::StringRef name) {
  return static_cast<bool>(module.lookupSymbol<mlir::func::FuncOp>(name));
}

/*void replaceWithApplyGate(OpBuilder &builder, Operation *op, StringRef gateFn)
{ builder.setInsertionPoint(op);

  auto loc = op->getLoc();
  auto qubit = op->getOperand(0);
  auto module = op->getParentOfType<ModuleOp>();

  // Call gate matrix creator (e.g., @create_hadamard)
  auto gateFunc = SymbolRefAttr::get(builder.getContext(), gateFn);
  auto gateCall = builder.create<func::CallOp>(loc,
builder.getType<TensorType>(), gateFunc, ValueRange{});

  // Call @apply_gate with (matrix, qubit)
  auto applyFunc = SymbolRefAttr::get(builder.getContext(), "apply_gate");
  auto applyCall = builder.create<func::CallOp>(loc,
builder.getType<TensorType>(), applyFunc, ValueRange{gateCall.getResult(0),
qubit});

  // Replace original quake op
  op->replaceAllUsesWith(applyCall.getResult(0));
  op->erase();
}

void replaceTwoQubitGate(OpBuilder &builder, Operation *op, StringRef gateFn) {
  builder.setInsertionPoint(op);
  Location loc = op->getLoc();

  auto ctrl = op->getOperand(0);
  auto target = op->getOperand(1);

  // Call @create_cz or @create_cx
  auto callCreate = builder.create<func::CallOp>(
      loc, FlatSymbolRefAttr::get(builder.getContext(), gateFn),
      builder.getType<TensorType>(), ValueRange{}
  );

  // Call @apply_gate2(matrix, qubit1, qubit2)
  auto applyCall = builder.create<func::CallOp>(
      loc, FlatSymbolRefAttr::get(builder.getContext(), "apply_gate2"),
      builder.getType<TensorType>(), ValueRange{callCreate.getResult(0), ctrl,
target}
  );

  op->replaceAllUsesWith(applyCall.getResult(0));
  op->erase();
}
*/
void insertCreateHadamard(OpBuilder &builder, ModuleOp module) {
  if (hasFunc(module, "create_hadamard"))
    return;

  builder.setInsertionPointToStart(module.getBody());
  Location loc = builder.getUnknownLoc();

  auto f32Type = builder.getF32Type();
  auto mat2x2f32 = RankedTensorType::get({2, 2}, f32Type);
  auto funcType = builder.getFunctionType({}, {mat2x2f32});

  auto func = builder.create<func::FuncOp>(loc, "create_hadamard", funcType);
  func.setPrivate();
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value s = builder.create<arith::ConstantOp>(
      loc, builder.getF32Type(), builder.getFloatAttr(f32Type, 0.70710678f));
  Value neg_s = builder.create<arith::ConstantOp>(
      loc, builder.getF32Type(), builder.getFloatAttr(f32Type, -0.70710678f));

  // Flatten all 4 scalar elements in row-major order
  Value result = builder.create<tensor::FromElementsOp>(
      loc, mat2x2f32, ValueRange{s, s, s, neg_s});

  builder.create<func::ReturnOp>(loc, result);
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

    // Iterate through all functions and insert the required functions for each
    // gate
    module.walk([&](Operation *op) {
      // Get the function name
      // StringRef funcName = op.getName();
      // if (!(funcName.find(std::string(CUDAQ_PREFIX_FUNCTION)) !=
      //  std::string::npos))
      //  return; // do nothing if the function is not cudaq kernel
      if (auto quakeH = dyn_cast_or_null<quake::HOp>(op)) {
        insertCreateHadamard(builder, module); // ✅ pass correct type
      }
    });

    // Iterate through all functions
    /*for (auto func : module.getOps<func::FuncOp>()) {
      func.walk([&](Operation *op) {
        if (auto quakeH = dyn_cast_or_null<quake::HOp>(op)) {
          replaceWithApplyGate(builder, quakeH, "create_hadamard");
        } else if (auto quakeZ = dyn_cast_or_null<quake::ZOp>(op)) {
          replaceWithApplyGate(builder, quakeZ, "create_z");
        }
        // Add other gate handlers as needed
      });
    }*/
  }

private:
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQuakeToLinAlgPass() {
  return std::make_unique<QuakeToLinAlg>();
}
