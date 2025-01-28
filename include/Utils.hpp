#ifndef UTILS_H
#define UTILS_H

#pragma once

using namespace mlir;

namespace mqss::utils{
 
  inline mlir::Value createFloatValue(mlir::OpBuilder &builder, mlir::Location loc, double value) {
  // Determine the floating-point type. This example assumes 32-bit float.
  auto floatType = builder.getF32Type();
  // Create a FloatAttr to hold the constant value.
  auto floatAttr = builder.getFloatAttr(floatType, value);
  // Use the builder to create an arith.constant operation.
  return builder.create<mlir::arith::ConstantOp>(loc, floatType, floatAttr).getResult();
  }
 
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

  inline mlir::Operation *getPreviousOperationOnTarget(mlir::Operation *currentOp, mlir::Value targetQubit){
    // Start from the previous operation
    mlir::Operation *prevOp = currentOp->getPrevNode();
    // Iterate through the previous operations in the block
    while (prevOp) {
      // Check if the operation has a target qubit and matches the given target
      if (auto quakeOp = dyn_cast<quake::OperatorInterface>(prevOp)) {
          int targetQCurr = mqss::utils::extractIndexFromQuakeExtractRefOp(targetQubit.getDefiningOp());
        for (mlir::Value target : quakeOp.getTargets()) {
          int targetQPrev = mqss::utils::extractIndexFromQuakeExtractRefOp(target.getDefiningOp());
          if (targetQCurr  == targetQPrev)
            return prevOp;
        }
        for (mlir::Value control : quakeOp.getControls()) {
          int controlQPrev = mqss::utils::extractIndexFromQuakeExtractRefOp(control.getDefiningOp());
          if (targetQCurr  == controlQPrev)
            return prevOp;
        }
      }
      // Move to the previous operation
      prevOp = prevOp->getPrevNode();
      }
      return nullptr; // No matching previous operation found
  }

  inline mlir::Operation *getNextOperationOnTarget(mlir::Operation *currentOp, mlir::Value targetQubit){
    // Start from the next operation
    mlir::Operation *nextOp = currentOp->getNextNode();
    // Iterate through the previous operations in the block
    while (nextOp) {
      // Check if the operation has a target qubit and matches the given target
      if (auto quakeOp = dyn_cast<quake::OperatorInterface>(nextOp)) {
          int targetQCurr = mqss::utils::extractIndexFromQuakeExtractRefOp(targetQubit.getDefiningOp());
        for (mlir::Value target : quakeOp.getTargets()) {
          int targetQNext = mqss::utils::extractIndexFromQuakeExtractRefOp(target.getDefiningOp());
          if (targetQCurr  == targetQNext)
            return nextOp;
        }
        for (mlir::Value control : quakeOp.getControls()) {
          int controlQNext = mqss::utils::extractIndexFromQuakeExtractRefOp(control.getDefiningOp());
          if (targetQCurr  == controlQNext)
            return nextOp;
        }
      }
      // Move to the previous operation
      nextOp = nextOp->getNextNode();
      }
      return nullptr; // No matching previous operation found
  }

  // Finds the pattern composed of T2, T1 and commute them to T1, T2
  template <typename T1, typename T2>
  inline void commuteOperation(mlir::Operation *currentOp,
                               int nCtrlsOp1,
                               int nTgtsOp1,
                               int nCtrlsOp2,
                               int nTgtsOp2){
    auto currentGate = dyn_cast_or_null<T2>(*currentOp);
    if (!currentGate)
      return;
    // check that the current gate is compliant with the number of controls and targets
    if (currentGate.getControls().size() != nCtrlsOp2 ||
        currentGate.getTargets().size() != nTgtsOp2)
      return;
    // get the previous operation to check the swap pattern
    auto prevOp = getPreviousOperationOnTarget(currentGate, currentGate.getTargets()[0]);
    if (!prevOp) return;
    auto previousGate = dyn_cast_or_null<T1>(prevOp);
    if (!previousGate)
      return;
    // check that the previous gate is compliant with the number of controls and targets
    if (previousGate.getControls().size() != nCtrlsOp1 ||
        previousGate.getTargets().size() != nTgtsOp1)
      return;  // check both targets are the same
    int targetPrev = extractIndexFromQuakeExtractRefOp(previousGate.getTargets()[0].getDefiningOp());
    int targetCurr = extractIndexFromQuakeExtractRefOp(currentGate.getTargets()[0].getDefiningOp());
    if (targetPrev != targetCurr)
      return;
    #ifdef DEBUG
      llvm::outs() << "Current Operation: ";
      currentGate->print(llvm::outs());
      llvm::outs() << "\n";
      llvm::outs() << "Previous Operation: ";
      previousGate->print(llvm::outs());
      llvm::outs() << "\n";
    #endif
    // At this point, I should de able to do the commutation
    // Swap the two operations by cloning them in reverse order.
    mlir::IRRewriter rewriter(currentGate->getContext());
    rewriter.setInsertionPointAfter(currentGate);
    auto newGate = rewriter.create<T1>(previousGate.getLoc(),
                                       previousGate.isAdj(),
                                       previousGate.getParameters(),
                                       previousGate.getControls(),
                                       previousGate.getTargets());
    rewriter.setInsertionPoint(newGate);
    rewriter.create<T2>(currentGate.getLoc(), currentGate.isAdj(),
                        currentGate.getParameters(), currentGate.getControls(),
                        currentGate.getTargets());
    // Erase the original operations
    rewriter.eraseOp(currentGate);
    rewriter.eraseOp(previousGate);
  }

  // Finds the pattern composed of T2, T1 and remove them
  // and T1 and T2 share the same control
  // Targets and controls should be the same on boths
  template <typename T1, typename T2>
  inline void patternCancellation(mlir::Operation *currentOp,
                               int nCtrlsOp1,
                               int nTgtsOp1,
                               int nCtrlsOp2,
                               int nTgtsOp2){
    auto currentGate = dyn_cast_or_null<T2>(*currentOp);
    if (!currentGate)
      return;
    // check that the current gate is compliant with the number of controls and targets
    if (currentGate.getControls().size() != nCtrlsOp2 ||
        currentGate.getTargets().size() != nTgtsOp2)
      return;
    // get the previous operation to check the swap pattern
    auto prevOp = getPreviousOperationOnTarget(currentGate, currentGate.getTargets()[0]);
    if(!prevOp) return;
    auto previousGate = dyn_cast_or_null<T1>(prevOp);
    if (!previousGate)
      return;
    // check that the previous gate is compliant with the number of controls and targets
    if (previousGate.getControls().size() != nCtrlsOp1 ||
        previousGate.getTargets().size() != nTgtsOp1)
      return;
    // check that targets and controls are the same!
    // At the moment I am checking all controls and all targets!
    if(currentGate.getControls().size() == previousGate.getControls().size()){
      std::vector<int> controlsCurr = getIndicesOfValueRange(currentGate.getControls());
      std::vector<int> controlsPrev = getIndicesOfValueRange(previousGate.getControls());
      // sort both arrays
      std::sort(controlsCurr.begin(), controlsCurr.end(), std::greater<int>());
      std::sort(controlsPrev.begin(), controlsPrev.end(), std::greater<int>());
      // compare both arrays
      if (!(std::equal(controlsCurr.begin(), controlsCurr.end(), controlsPrev.begin())))
        return;
    } else return;
    // so far, controls are the same, now check the targets
    if(currentGate.getTargets().size() == previousGate.getTargets().size()){
      std::vector<int> targetsCurr = getIndicesOfValueRange(currentGate.getTargets());
      std::vector<int> targetsPrev = getIndicesOfValueRange(previousGate.getTargets());
      // sort both arrays
      std::sort(targetsCurr.begin(), targetsCurr.end(), std::greater<int>());
      std::sort(targetsPrev.begin(), targetsPrev.end(), std::greater<int>());
      // compare both arrays
      if (!(std::equal(targetsCurr.begin(), targetsCurr.end(), targetsPrev.begin())))
        return;
    } else return;
    #ifdef DEBUG
      llvm::outs() << "Current Operation: ";
      currentGate->print(llvm::outs());
      llvm::outs() << "\n";
      llvm::outs() << "Previous Operation: ";
      previousGate->print(llvm::outs());
      llvm::outs() << "\n";
    #endif
    // At this point, I should de able to remove the pattern
    mlir::IRRewriter rewriter(currentGate->getContext());
    // Erase the operations
    rewriter.eraseOp(currentGate);
    rewriter.eraseOp(previousGate);
  }

  // Finds the pattern composed of T1, T2 and switches them
  // and assigns the types T3, and T4
  // Targets and controls should be the same on boths
  // this only works at the moment for single qubit gates
  template <typename T1, typename T2, typename T3, typename T4>
  inline void patternSwitch(mlir::Operation *currentOp){
    auto currentGate = dyn_cast_or_null<T2>(*currentOp);
    if (!currentGate)
      return;
    // check single qubit T2 operation
    if(currentGate.getControls().size()!=0 ||
       currentGate.getTargets().size()!=1)
    return;
    // get previous
    auto prevOp = getPreviousOperationOnTarget(currentGate, currentGate.getTargets()[0]);
    if(!prevOp) return;
    auto prevGate = dyn_cast<T1>(prevOp);
    // check single qubit operation
    if(prevGate.getControls().size()!=0 ||
       prevGate.getTargets().size()!=1)
      return;
    // I found the pattern, then I remove it from the circuit
    #ifdef DEBUG
      llvm::outs() << "Current Operation: ";
      currentGate->print(llvm::outs());
      llvm::outs() << "\n";
      llvm::outs() << "Previous Operation: ";
      prevGate->print(llvm::outs());
      llvm::outs() << "\n";
    #endif
    mlir::IRRewriter rewriter(currentGate->getContext());
    rewriter.setInsertionPointAfter(currentGate);
    auto newGate = rewriter.create<T3>(currentGate.getLoc(),
                                       currentGate.isAdj(),
                                       currentGate.getParameters(),
                                       currentGate.getControls(),
                                       currentGate.getTargets());
    rewriter.setInsertionPointAfter(newGate);
    rewriter.create<T4>(prevGate.getLoc(),
                        prevGate.isAdj(),
                        prevGate.getParameters(),
                        prevGate.getControls(),
                        prevGate.getTargets());
    rewriter.eraseOp(prevGate);
    rewriter.eraseOp(currentGate);
  }
} // end namespace
#endif // UTILS_H
