#ifndef UTILS_H
#define UTILS_H

#pragma once

using namespace mlir;

namespace mqss::utils{
  
  inline double extractDoubleArgumentValue(mlir::Operation *op){
    if (auto constantOp = dyn_cast<mlir::arith::ConstantOp>(op))
      if (auto floatAttr = constantOp.getValue().dyn_cast<mlir::FloatAttr>())
        return static_cast<float>(floatAttr.getValueAsDouble());
    return -1.0;
  }
  
  inline int64_t extractIndexFromQuakeExtractRefOp(mlir::Operation *op) {
    if (auto extractRefOp = llvm::dyn_cast<quake::ExtractRefOp>(op)) {
      auto rawIndexAttr = extractRefOp->getAttrOfType<mlir::IntegerAttr>("rawIndex");
      return rawIndexAttr.getInt();
    }
    return -1;
  }
  // function to get the number of qubits in a given quantum kernel
  inline int getNumberOfQubits(func::FuncOp circuit){
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
  // Function to get the number of classical bits allocated in a given quantum kernel, it also stores information of the qiubit position
  inline int getNumberOfClassicalBits(func::FuncOp circuit, std::map<int, int> &measurements){
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
  // Function to get the number of classical bits allocated in a given quantum kernel
  inline int getNumberOfClassicalBits(func::FuncOp circuit){
    int numBits=0;
    circuit.walk([&](mlir::Operation *op) {
      if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)){
        for (auto operand : op->getOperands()) {
          if (operand.getType().isa<quake::RefType>()) { // Check if it's a qubit reference
            int qubitIndex = extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
            if (qubitIndex == -1)
              throw std::runtime_error("Non valid qubit index for measurement!");
            numBits += 1;
          }else if (operand.getType().isa<quake::VeqType>()) {
            auto qvecType = operand.getType().dyn_cast<quake::VeqType>();
            numBits += qvecType.getSize();
          }
        }
      }
    });
    return numBits;
  }

  inline std::vector<int> getIndicesOfValueRange(mlir::ValueRange array){
    std::vector<int> indices;
    for(auto value : array){
      int qubit_index = extractIndexFromQuakeExtractRefOp(value.getDefiningOp());
      indices.push_back(qubit_index);
    }
    return indices;
  }

  // At the moment, it is assumed that the parameters are of type Double
  inline std::vector<double> getParametersValues(mlir::ValueRange array){
    std::vector<double> parameters;
    for(auto value : array){
      double param = extractDoubleArgumentValue(value.getDefiningOp());
      parameters.push_back(param);
    }
    return parameters;
  }


} // end namespace
#endif // UTILS_H
