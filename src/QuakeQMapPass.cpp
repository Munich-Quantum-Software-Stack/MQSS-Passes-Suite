#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Passes.hpp"
#include "qdmi.h"
#include "sc/heuristic/HeuristicMapper.hpp"

using namespace mlir;

namespace {

class QuakeQMapPass
    : public PassWrapper<QuakeQMapPass, OperationPass<func::FuncOp>> {
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
      #ifdef DEBUG
      llvm::errs() << "Operation ";
      op->print(llvm::errs());
      llvm::errs() <<"\n";
      llvm::errs() << "\tRotation with angle " << angle << " on qubit "<< qubit<<"\n";
      #endif
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
      if (op->getOperands().size() ==2){
        int qubit_ctrl,qubit_target;
        Value operand1 = op->getOperands()[0];
        qubit_ctrl = extractIndexFromQuakeExtractRefOp(operand1.getDefiningOp());
        Value operand2 = op->getOperands()[1];
        qubit_target = extractIndexFromQuakeExtractRefOp(operand2.getDefiningOp());
        #ifdef DEBUG
        llvm::errs() << "Operation ";
        op->print(llvm::errs());
        llvm::errs() <<"\n";
        llvm::errs() << "\tqubit_ctrl " << qubit_ctrl << " qubit_target "<< qubit_target <<"\n";
        #endif
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
        #ifdef DEBUG
        llvm::errs() << "Operation ";
        op->print(llvm::errs());
        llvm::errs() <<"\n";
        llvm::errs() << "\tSingle qubit operation on qubit " << qubit <<"\n";
        #endif
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
        #ifdef DEBUG
        llvm::errs() << "Operation ";
        op->print(llvm::errs());
        llvm::errs() <<"\n";
        llvm::errs() << "\tSingle qubit operation on qubit " << qubit <<"\n";
        #endif
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
      #ifdef DEBUG
      llvm::errs()<< "Operation ";
      op->print(llvm::errs());
      llvm::errs() <<"\n";
      #endif
      if (op->getOperands().size()!=1)
        throw std::runtime_error("ill-formed measurement gate!");
      Value operand = op->getOperands()[0];
      if (operand.getType().isa<quake::RefType>()) {
        int qubitIndex = extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
        if (qubitIndex == -1)
          throw std::runtime_error("Non valid qubit index for measurement!");
        qc.measure(static_cast<qc::Qubit>(qubitIndex),measurements.at(qubitIndex));
        #ifdef DEBUG
        llvm::errs() << "\tMeasurement on qubit index " <<qubitIndex << "\n";
        #endif
      }else if (operand.getType().isa<quake::VeqType>()) {          
        auto qvecType = operand.getType().dyn_cast<quake::VeqType>();
        int nQubits = qvecType.getSize();
        qc.measureAll();
        #ifdef DEBUG
        llvm::errs() << "\tMeasurement on vector of size " << nQubits << "\n";
        #endif
        // TODO: Here, I am measuring all the qubits, if slicing is allowed in cudaq, then I has to be implemented
        //for (std::size_t i = 0; i < nQubits; ++i) {
        //  qc.measure(static_cast<qc::Qubit>(i), i);
        //}
      }
    }
  }
  int getNumberOfQubits(func::FuncOp circuit){
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
  int getNumberOfClassicalBits(func::FuncOp circuit, std::map<int, int> &measurements){
    int numBits=0;
    circuit.walk([&](mlir::Operation *op) {
      if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)){
        for (auto operand : op->getOperands()) {
          if (operand.getType().isa<quake::RefType>()) { // Check if it's a qubit reference
            int qubitIndex = extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
            if (qubitIndex == -1)
              throw std::runtime_error("Non valid qubit index for measurement!");
            measurements[qubitIndex] = numBits;
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
  /*
      3
     / \
    4   2
    |   |
    0---1
  */
    const CouplingMap cm = {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3},
                          {3, 2}, {3, 4}, {4, 3}, {4, 0}, {0, 4}};
    arch.loadCouplingMap(5, cm);
    std::cout << "Dumping the architecture " << std::endl;
    Architecture::printCouplingMap(arch.getCouplingMap(), std::cout);
    auto circuit = getOperation();
    // Get the function name
    StringRef funcName = circuit.getName();
    if (!(funcName.find(std::string(CUDAQ_PREFIX_FUNCTION)) != std::string::npos))
      return; // do nothing if the funcion is not cudaq kernel
    llvm::outs() << "Kernel name: " << funcName << "\n";

    std::map<int, int> measurements; // key: qubit, value register index
    int numQubits = getNumberOfQubits(circuit);
    int numBits = getNumberOfClassicalBits(circuit,measurements);
    #ifdef DEBUG
    llvm::errs() <<"Number of input qubits " << numQubits << "\n";
    llvm::errs() <<"Number of output bits " << numBits << "\n";
    #endif
    auto qc = qc::QuantumComputation(numQubits, numBits);
    // loading rotation gates
    circuit.walk([&](mlir::Operation *op) {
      loadRotationGatesToQC(op,qc);
      loadXYZGatesToQC(op,qc);
      loadSTHGatesToQC(op,qc);
      loadMeasurementsToQC(op,qc,measurements);
    });

    llvm::errs() << "Dumping QC:\n";
    //std::ostream ros(llvm::errs());
    qc.print(std::cout);
    // Map the circuit
    const auto mapper = std::make_unique<HeuristicMapper>(qc, arch);
    Configuration settings{};
    settings.heuristic = Heuristic::GateCountMaxDistance;
    settings.layering = Layering::DisjointQubits;
    settings.initialLayout = InitialLayout::Identity;
    settings.preMappingOptimizations = false;
    settings.postMappingOptimizations = false;
    settings.lookaheadHeuristic = LookaheadHeuristic::None;
//    settings.debug = false;
    settings.addMeasurementsToMappedCircuit = true;
    mapper->map(settings);
    // TODO: There should be other way to get the mapped circuit
    //        do not like to down the mapped circuit to QASM and
    //        then back to qc
    auto qcMapped = qc::QuantumComputation();
    std::stringstream qasm{};
    mapper->dumpResult(qasm, qc::Format::OpenQASM3);
    qcMapped.import(qasm, qc::Format::OpenQASM3);
    // cleaning the mlir::funcOp corresponding to the quake circuit
    for (auto &block : circuit.getBody()) {
      block.clear();  // Clears all operations in the current block
    } 
    OpBuilder builder(&circuit.getBody());
    Location loc = circuit.getLoc();
    // allocate the qubits
    Value qubits =
      builder.create<quake::AllocaOp>(circuit.getLoc(),quake::VeqType::get(builder.getContext(), numQubits));
    DenseMap<std::size_t, Value> finalQubitWire;
    // then traverse the mapped QuantumComputation and annotate it in the
    // mlir func
    for (const auto& op : qcMapped){
      auto &targets  = op->getTargets();
      auto &controls = op->getControls();
      if (targets.size()==2 && controls.size()==0){
        if(op->getType() == qc::SWAP){
          // extract the reference of the first target qubit
          auto target1Ref = builder.create<quake::ExtractRefOp>(loc, qubits, targets[0]);
          auto target2Ref = builder.create<quake::ExtractRefOp>(loc, qubits, targets[1]);
          SmallVector<Value> ctrls = {};
          SmallVector<Value> targets = {};
          targets.push_back(target1Ref);
          targets.push_back(target2Ref);
          builder.create<quake::SwapOp>(loc,ctrls,targets);
        }
      }
      if (targets.size() == 1 && controls.size() == 1){
        // extract the reference of the control qubit
        auto controlRef = builder.create<quake::ExtractRefOp>(loc, qubits, controls.begin()->qubit);
        // extract the reference of the target qubit
        auto targetRef = builder.create<quake::ExtractRefOp>(loc, qubits, targets[0]);
        //create the XOp operation
        switch (op->getType()) {
          case qc::X:
            builder.create<quake::XOp>(loc, controlRef.getResult(), targetRef.getResult());
          break;
          case qc::Y:
            builder.create<quake::YOp>(loc, controlRef.getResult(), targetRef.getResult());
          break;
          case qc::Z:
            builder.create<quake::ZOp>(loc, controlRef.getResult(), targetRef.getResult());
          break;
          case qc::SWAP:
            llvm::errs() << "SWAP gate from " << targets[0] << " to " << targets[1] << "\n";
          break; 
        }
      }
      if (targets.size() == 1 && controls.size() == 0){
        // single qubit functions
        auto &targets  = op->getTargets();
        // extract the reference of the target qubit
        auto targetRef = builder.create<quake::ExtractRefOp>(loc, qubits, targets[0]);
        switch (op->getType()) {
          case qc::X:
            builder.create<quake::XOp>(loc, targetRef.getResult());
          break;
          case qc::Y:
            builder.create<quake::YOp>(loc, targetRef.getResult());
          break;
          case qc::Z:
            builder.create<quake::ZOp>(loc, targetRef.getResult());
          break;
          case qc::H:
            builder.create<quake::HOp>(loc, targetRef.getResult());
          break;
          case qc::S:
            builder.create<quake::SOp>(loc, targetRef.getResult());
          break;
          case qc::T:
            builder.create<quake::TOp>(loc, targetRef.getResult());
          break;
          case qc::Measure:
            Type measTy = quake::MeasureType::get(builder.getContext());
            builder.create<quake::MzOp>(loc, measTy, targetRef.getResult()).getMeasOut();
          break;
        }
      }
    }
    builder.create<func::ReturnOp>(circuit.getLoc());
    std::cout << "Dumping QC after mapping:\n";
    //std::ostream ros(llvm::errs());
    qcMapped.print(std::cout);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mqss::opt::createQuakeQMapPass(llvm::raw_string_ostream &ostream){
  return std::make_unique<QuakeQMapPass>(ostream);
}
