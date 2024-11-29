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
