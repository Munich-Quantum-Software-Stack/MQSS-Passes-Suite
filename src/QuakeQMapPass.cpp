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
  // loading rotation gates
  void loadRotationGatesToQC(Operation *op, qc::QuantumComputation &qc){
    if (isa<quake::RxOp>(op) || isa<quake::RyOp>(op) || isa<quake::RzOp>(op)){
      int qubit = -1;
      double angle = -1.0;
      if (op->getOperands().size()!=2)
        throw std::runtime_error("ill-formed rotation gate!");
      Value operand1 = op->getOperands()[0];
      angle = extractDoubleArgumentValue(operand1.getDefiningOp());
      Value operand2 = op->getOperands()[1];
      qubit= extractIndexFromQuakeExtractRefOp(operand2.getDefiningOp());
      outputStream << "angle " << angle << " qubit "<< qubit<<"\n";
      if(angle == -1.0 || qubit == -1)
        throw std::runtime_error("ill-formed rotation gate!");
      if (isa<quake::RxOp>(op))
        qc.rx(angle,qubit);
      if (isa<quake::RyOp>(op))
        qc.ry(angle,qubit);
      if (isa<quake::RzOp>(op))
        qc.rz(angle,qubit);
    }
  }
  // loading X, Y , Z
  // two bits X,Y and Z refers to controlled Cx, Cy, and Cz
  // single bits are just x,y,and z
  void loadXYZGatesToQC(Operation *op, qc::QuantumComputation &qc){
    if (isa<quake::XOp>(op) || isa<quake::YOp>(op) || isa<quake::ZOp>(op)){
      // controlled operations
      int qubit_ctrl,qubit_target;
      if (op->getOperands().size() ==2){
        Value operand1 = op->getOperands()[0];
        qubit_ctrl = extractIndexFromQuakeExtractRefOp(operand1.getDefiningOp());
        Value operand2 = op->getOperands()[1];
        qubit_target = extractIndexFromQuakeExtractRefOp(operand2.getDefiningOp());
        outputStream << "qubit_ctrl " << qubit_ctrl << " qubit_target "<< qubit_target <<"\n";
        if(qubit_ctrl == -1 || qubit_target == -1)
          throw std::runtime_error("ill-formed controlled gate!");
        if (isa<quake::XOp>(op))
          qc.cx(qubit_ctrl,qubit_target);
        if (isa<quake::YOp>(op))
          qc.cy(qubit_ctrl,qubit_target);
        if (isa<quake::ZOp>(op))
          qc.cz(qubit_ctrl,qubit_target);
      }
      // single qubit operations
      if (op->getOperands().size() ==1){
        Value operand1 = op->getOperands()[0];
        int qubit=extractIndexFromQuakeExtractRefOp(operand1.getDefiningOp());
        outputStream << "Operation ";
        op->print(outputStream);
        outputStream <<"\n";
        outputStream << "Single qubit operation qubit " << qubit <<"\n";
        if(qubit == -1)
          throw std::runtime_error("ill-formed single gate X, Y and Z!");
        if (isa<quake::XOp>(op))
          qc.x(qubit);
        if (isa<quake::YOp>(op))
          qc.y(qubit);
        if (isa<quake::ZOp>(op))
          qc.z(qubit);
      }
    }
  }
  // loading S,T,H single qubit gates
  void loadSTHGatesToQC(Operation *op, qc::QuantumComputation &qc){
    if (isa<quake::SOp>(op) || isa<quake::TOp>(op) || isa<quake::HOp>(op)){
      int qubit_ctrl,qubit_target;
      // single qubit operations
      if (op->getOperands().size() ==1){
        Value operand1 = op->getOperands()[0];
        int qubit=extractIndexFromQuakeExtractRefOp(operand1.getDefiningOp());
        outputStream << "Operation ";
        op->print(outputStream);
        outputStream <<"\n";
        outputStream << "Single qubit operation qubit " << qubit <<"\n";
        if(qubit == -1)
          throw std::runtime_error("ill-formed single gate, S, T or H !");
        if (isa<quake::SOp>(op))
          qc.s(qubit);
        if (isa<quake::TOp>(op))
          qc.t(qubit);
        if (isa<quake::HOp>(op))
          qc.h(qubit);
      }
    }
  }
  // loading measurements
  void loadMeasurementsToQC(Operation *op, qc::QuantumComputation &qc,std::map<int,int> measurements){
    int qubit=-1, result =-1;
    if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)){
      outputStream << "size" << op->getOperands().size() << "\n";
      outputStream << "Operation ";
      op->print(outputStream);
      outputStream <<"\n";
      if (op->getOperands().size()!=1)
        throw std::runtime_error("ill-formed measurement gate!");
      Value operand = op->getOperands()[0];
      if (operand.getType().isa<quake::RefType>()) {
        int qubitIndex = extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
        if (qubitIndex == -1)
          throw std::runtime_error("Non valid qubit index for measurement!");
        qc.measure(static_cast<qc::Qubit>(qubitIndex),measurements.at(qubitIndex));
        outputStream << "Measurement on qubit index" <<qubitIndex << "\n";
      }else if (operand.getType().isa<quake::VeqType>()) {          
        outputStream << "Measurement on all the vector \n";
        auto qvecType = operand.getType().dyn_cast<quake::VeqType>();
        int nQubits = qvecType.getSize();
        qc.measureAll();
        // TODO: Here, I am measuring all the qubits, if slicing is allowed in cudaq, then I has to be implemented
        //for (std::size_t i = 0; i < nQubits; ++i) {
        //  qc.measure(static_cast<qc::Qubit>(i), i);
        //}
      }
    }
  }
  int getNumberOfQubits(ModuleOp circuit){
    int numQubits = 0;
    circuit.walk([&](quake::AllocaOp allocOp) {
      if (auto qrefType = allocOp.getType().dyn_cast<quake::RefType>()) {
        numQubits += 1;
      } else if (auto qvecType = allocOp.getType().dyn_cast<quake::VeqType>()) {
        numQubits += qvecType.getSize();
      }
    });
    return numQubits;
  }
  int getNumberOfClassicalBits(ModuleOp circuit, std::map<int, int> &measurements){
    int numBits=0;
    circuit.walk([&](mlir::Operation *op) {
      if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)){
        for (auto operand : op->getOperands()) {
          if (operand.getType().isa<quake::RefType>()) { // Check if it's a qubit reference
            int qubitIndex = extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
            if (qubitIndex == -1)
              throw std::runtime_error("Non valid qubit index for measurement!");            measurements[qubitIndex] = numBits;
            numBits += 1;
          }else if (operand.getType().isa<quake::VeqType>()) {
            auto qvecType = operand.getType().dyn_cast<quake::VeqType>();
            numBits += qvecType.getSize();
            for (int i=0; i<numBits; i++){
              measurements[i]=i;
            }
          }
        }
      }
    });
    return numBits;
  }


public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuakeQMapPass)

  QuakeQMapPass(llvm::raw_string_ostream &ostream) : outputStream(ostream) {
    //architecture = mqt::createArchitecture(dev);
  }

  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

  void runOnOperation() override {
    Architecture arch{};
    arch.loadCouplingMap(AvailableArchitecture::IbmqLondon);

    auto circuit = getOperation();
    std::map<int, int> measurements; // key: qubit, value register index
    int numQubits = getNumberOfQubits(circuit);
    int numBits = getNumberOfClassicalBits(circuit,measurements);
    outputStream<<"Number of input qubits " << numQubits << "\n";
    outputStream<<"Number of output bits " << numBits << "\n";
    auto qc = qc::QuantumComputation(numQubits, numBits);
    // loading rotation gates
    circuit.walk([&](mlir::Operation *op) {
      loadRotationGatesToQC(op,qc);
      loadXYZGatesToQC(op,qc);
      loadSTHGatesToQC(op,qc);
      loadMeasurementsToQC(op,qc,measurements);
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQuakeQMapPass(llvm::raw_string_ostream &ostream){
  return std::make_unique<QuakeQMapPass>(ostream);
}
