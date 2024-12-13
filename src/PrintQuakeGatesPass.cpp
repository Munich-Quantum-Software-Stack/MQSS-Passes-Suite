/* This code and any associated documentation is provided "as is"

 IN NO EVENT SHALL LEIBNIZ-RECHENZENTRUM (LRZ) BE LIABLE TO ANY PARTY FOR
 DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
 OF THE USE OF THIS CODE AND ITS DOCUMENTATION, EVEN IF LEIBNIZ-RECHENZENTRUM
 (LRZ) HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 THE AFOREMENTIONED EXCLUSIONS OF LIABILITY DO NOT APPLY IN CASE OF INTENT
 BY LEIBNIZ-RECHENZENTRUM (LRZ).

 LEIBNIZ-RECHENZENTRUM (LRZ), SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE.

 THE CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, LEIBNIZ-RECHENZENTRUM (LRZ)
 HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 MODIFICATIONS.
-------------------------------------------------------------------------
  @author Martin Letras
  @date   December 2024
  @version 1.0
  @ brief
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
#include "Passes.hpp"

using namespace mlir;

namespace {

class PrintQuakeGatesPass
    : public PassWrapper<PrintQuakeGatesPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintQuakeGatesPass)

  PrintQuakeGatesPass(llvm::raw_string_ostream &ostream) : outputStream(ostream) {}


  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op){
      if (op->getDialect()->getNamespace() == "quake") {
        outputStream << "Quantum Operation: " << op->getName().getStringRef() << "\n";

        // Iterate over the operands (qubits) the operation acts on
        for (Value operand : op->getOperands()) {
          if (operand.getType().isa<quake::RefType>()) { // Check if it's a qubit reference
            outputStream << "  Acts on qubit: " << operand << "\n";
          }
        }

      }
    });
  }
private:
  llvm::raw_string_ostream &outputStream; // Store the output stream
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createPrintQuakeGatesPass(llvm::raw_string_ostream &ostream){
  return std::make_unique<PrintQuakeGatesPass>(ostream);
}
