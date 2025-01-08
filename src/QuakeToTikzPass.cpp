/* This code and any associated documentation is provided "as is"

Copyright 2024 Munich Quantum Software Stack Project

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
  date   Januray 2025
  version 1.0
  brief
  PrintQuakeGatesPass(llvm::raw_string_ostream ostream)
  Example MLIR pass that shows how to traverse a Quantum kernel written in
  QUAKE MLIR.
  The pass prints in ostream the type of each quantum gate and its operand(s)
  qubits.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Passes.hpp"
#include "Utils.hpp"

using namespace mlir;

void dumpQuakeOperation(mlir::Operation *op, std::vector<std::vector<std::string>> &qubitLines){
  if (op->getDialect()->getNamespace() != "quake")
    return; // do nothing if it is not a quake operation
  if(isa<quake::AllocaOp>(op) || isa<quake::ExtractRefOp>(op)) 
    return; // do nothing if the next operation is a qubit allocation
  StringRef gateName          = op->getName().getStringRef();
  llvm::outs() << "Operation: " << gateName << "\n";
  if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op))
    return; // 
  auto gate = dyn_cast<quake::OperatorInterface>(op);
  mlir::ValueRange parameters = gate.getParameters();
  mlir::ValueRange targets    = gate.getTargets();
  mlir::ValueRange controls   = gate.getControls();
  llvm::outs() << "\tParameters: "  << parameters.size() << " Targets: " << targets.size() << " Controls :" << controls.size() << "\n";
  // at the moment just add the gate
  if(isa<quake::SwapOp>(op))
    return; // special handle for swap because it has multiple targets
  if (targets.size() != 1)
    throw std::runtime_error("At this point only single target gates are expected !!");
  int target_qubit = mqss::utils::extractIndexFromQuakeExtractRefOp(targets[0].getDefiningOp());
  qubitLines[target_qubit].push_back("\\gate{"+std::string(gateName)+"}");
} 
  
  namespace {
  
  /// Conversion pattern for Quake operations to Quantikz LaTeX.
  /*class QuakeToTikzPattern : public ConversionPattern {
  public:
      explicit QuakeToTikzPattern(MLIRContext *context)
          : ConversionPattern("quake.op_name", 1, context) {}
  
      LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const override {
          if (auto hOp = dyn_cast<quake::HOp>(op)) {
              // Handle H gate
              llvm::outs() << "\\gate{H} & ";
              return success();
          } else if (auto xOp = dyn_cast<quake::XOp>(op)) {
              // Handle CX gate
              llvm::outs() << "\\ctrl{" << operands[0] << "} & \\targ{" << operands[1] << "} \\\\\n";
              return success();
          } else if (auto mzOp = dyn_cast<quake::MzOp>(op)) {
              // Handle Measurement
              llvm::outs() << "\\meter & \\qw \\\\\n";
              return success();
          }
  
          return failure();
      }
  };*/
  
  class QuakeToTikzPass
      : public PassWrapper<QuakeToTikzPass, OperationPass<func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuakeToTikzPass)
  
    QuakeToTikzPass(llvm::raw_string_ostream &ostream) : outputStream(ostream) {}
  
    llvm::StringRef getArgument() const override { return "convert-quake-to-tikz"; }
    llvm::StringRef getDescription() const override { return "Lower Quake Operations to LaTeX TiKz"; };
  
    void runOnOperation() override {
      auto circuit = getOperation();
      
      /*MLIRContext *context = &getContext();
  
      // Conversion target
      ConversionTarget target(*context);
      //target.addLegalDialect<StandardOpsDialect>();
      //target.addIllegalDialect<quake::QuakeDialect>();
  
      // Define rewrite patterns
      RewritePatternSet patterns(context);
    patterns.add<QuakeToTikzPattern>(context);

    // Apply conversion
    if (failed(applyPartialConversion(circuit, target, std::move(patterns)))) {
        signalPassFailure();
    }*/
    // Get the function name
    StringRef funcName = circuit.getName();
    if (!(funcName.find(std::string(CUDAQ_PREFIX_FUNCTION)) != std::string::npos))
      return; // do nothing if the funcion is not cudaq kernel

    std::map<int, int> measurements; // key: qubit, value register index   
    int numQubits = mqss::utils::getNumberOfQubits(circuit);
    int numBits   = mqss::utils::getNumberOfClassicalBits(circuit,measurements);

    std::vector<std::vector<std::string>> qubitTikz(numQubits);
    for(int i=0; i < numQubits; i++)
      qubitTikz[i].push_back({"\\lstick{\\ket{0}}"});
    circuit.walk([&](Operation *op){
      dumpQuakeOperation(op,qubitTikz);
    });
    llvm::outs() << "Num qubits " << numQubits << "\n";
    // dumping the lines into the ostream
    outputStream << "\\begin{quantikz}\n";
    int countQubit = 0;
    for(auto qubit  : qubitTikz){
      int size = qubit.size();
      llvm::outs() << "Ops size " << size << "\n";
      int countOps = 0;
      outputStream << "\t";
      for(auto op : qubit){
        outputStream << op;
        if (countOps != size - 1)
          outputStream << "\t&\t";
        countOps++;
      }
      if (countQubit != numQubits-1)
        outputStream << "\\\\\n";
      else
        outputStream << "\n";
      countQubit++;
    }
    outputStream << "\\end{quantikz}";
  }
private:
  llvm::raw_string_ostream &outputStream; // Store the tikz circuit
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQuakeToTikzPass(llvm::raw_string_ostream &ostream){
  return std::make_unique<QuakeToTikzPass>(ostream);
}
