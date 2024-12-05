#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

#include "Passes.hpp"
#include "qdmi.h"
//#include "ArchitectureFactory.hpp"
#include "sc/heuristic/HeuristicMapper.hpp"

using namespace mlir;

namespace {

class QuakeQMapPass
    : public PassWrapper<QuakeQMapPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuakeQMapPass)

  QuakeQMapPass(llvm::raw_string_ostream &ostream) : outputStream(ostream) {
    //architecture = mqt::createArchitecture(dev);
  }

  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

  void runOnOperation() override {
    Architecture arch{};
    arch.loadCouplingMap(AvailableArchitecture::IbmqLondon);

    int numQubits = 0;
    int numBits = 0;
    auto circuit = getOperation();
    // getting the number of qubits
    circuit.walk([&](quake::AllocaOp allocOp) {
      if (auto qrefType = allocOp.getType().dyn_cast<quake::RefType>()) {
        numQubits += 1;
      } else if (auto qvecType = allocOp.getType().dyn_cast<quake::VeqType>()) {
        numQubits += qvecType.getSize();
      }
    });
    outputStream<<"Number of input qubits " << numQubits << "\n";
    // getting the number of classical bits required to read the output
    circuit.walk([&](mlir::Operation *op) {
      if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)){
        for (auto operand : op->getOperands()) {
          if (operand.getType().isa<quake::RefType>()) { // Check if it's a qubit reference
            numBits += 1;
          }else if (operand.getType().isa<quake::VeqType>()) {
            auto qvecType = operand.getType().dyn_cast<quake::VeqType>();
            numBits += qvecType.getSize();
          }
        }
      }
    });
    outputStream<<"Number of output bits " << numBits << "\n";
    auto qc = qc::QuantumComputation(numQubits, numBits);
    // loading rotation gates
    circuit.walk([&](mlir::Operation *op) {
        int qubit = -1;
        double angle = -1.0;
      if (isa<quake::RxOp>(op) || isa<quake::RyOp>(op) || isa<quake::RzOp>(op)){
        op->getOperands().size(); 
        if (op->getOperands().size()!=2)
          std::runtime_error("ill-formed rotation gate!");
        Value operand1 = op->getOperands()[0];
        angle = extractDoubleArgumentValue(operand1.getDefiningOp());
        Value operand2 = op->getOperands()[1];
        qubit= extractIndexFromQuakeExtractRefOp(operand2.getDefiningOp());
        outputStream << "angle " << angle << " qubit "<< qubit<<"\n";
      }
      if(angle == -1.0 || qubit == -1) 
        std::runtime_error("ill-formed rotation gate!");
      if (isa<quake::RxOp>(op))
        qc.rx(angle,qubit);
      if (isa<quake::RyOp>(op))
        qc.ry(angle,qubit);
      if (isa<quake::RzOp>(op))
        qc.rz(angle,qubit);
    });
    // loading controlled gates
    circuit.walk([&](mlir::Operation *op) {
      int qubit_ctrl,qubit_target;
      if (isa<quake::XOp>(op) || isa<quake::YOp>(op) || isa<quake::ZOp>(op)){
        op->getOperands().size();
        if (op->getOperands().size()!=2)
          std::runtime_error("ill-formed rotation gate!");
        Value operand1 = op->getOperands()[0];
        qubit_ctrl = extractIndexFromQuakeExtractRefOp(operand1.getDefiningOp());
        Value operand2 = op->getOperands()[1];
        qubit_target = extractIndexFromQuakeExtractRefOp(operand2.getDefiningOp());
        outputStream << "qubit_ctrl " << qubit_ctrl << " qubit_target "<< qubit_target <<"\n";
      }
      if(qubit_ctrl == -1 || qubit_target == -1) 
        std::runtime_error("ill-formed rotation gate!");
      if (isa<quake::XOp>(op))
        qc.cx(qubit_ctrl,qubit_target);
      if (isa<quake::YOp>(op))
        qc.cy(qubit_ctrl,qubit_target);
      if (isa<quake::ZOp>(op))
        qc.cz(qubit_ctrl,qubit_target);
    });
  }



private:
  llvm::raw_string_ostream &outputStream; // Store the output stream
  //Architecture architecture;
double extractDoubleArgumentValue(mlir::Operation *op){
  if (auto constantOp = dyn_cast<mlir::arith::ConstantOp>(op))
    if (auto floatAttr = constantOp.getValue().dyn_cast<mlir::FloatAttr>())
      return static_cast<float>(floatAttr.getValueAsDouble()); 
  return -1.0;
}

int64_t extractIndexFromQuakeExtractRefOp(mlir::Operation *op) {
  if (auto extractRefOp = llvm::dyn_cast<quake::ExtractRefOp>(op)) {
    auto rawIndexAttr = extractRefOp->getAttrOfType<mlir::IntegerAttr>("rawIndex");
    return rawIndexAttr.getInt();
  }
  return -1;
}

};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQuakeQMapPass(llvm::raw_string_ostream &ostream){
  return std::make_unique<QuakeQMapPass>(ostream);
}
