/**
 * @file QirZExtTransform.cpp
 * @brief Implementation of the 'QirZExtTransformPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirZExtTransform.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from: This pass replacea Zext instruction of 1-bit integer with
 * select. i1 type is boolean. Instead of extending the value with Zext, we can
 * use the value as a condition to decide between 32-bit 1 or 0
 */

#include "../headers/QirZYZTransform.hpp"
//#include "../headers/utilities.hpp"
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */

std::vector<Value *> QirZYZTransformPass::getDecompositionAngles(LLVMContext &context,
                                            ComplexMatrix theGate) {

  double detArg = getTheAngle(det(theGate));
  double phase = 0.5 * detArg;
  double theta =
      2.0 * std::atan2(std::abs(theGate[0][1]), std::abs(theGate[0][0]));
  double ang1 = std::arg(theGate[1][1]);
  double ang2 = std::arg(theGate[1][0]);
  double phi = ang1 + ang2 - detArg;
  double lam = ang1 - ang2;

  Value *phiValue = ConstantFP::get(context, APFloat(static_cast<float>(phi)));
  Value *lamValue = ConstantFP::get(context, APFloat(static_cast<float>(lam)));
  Value *thetaValue =
      ConstantFP::get(context, APFloat(static_cast<float>(theta)));

  return {phiValue, lamValue, thetaValue};
}

PreservedAnalyses QirZYZTransformPass::run(Module &module,
                                           ModuleAnalysisManager & /*MAM*/) {
  std::string gatesToDecompose[4] = {
      "__quantum__qis__rx__body", "__quantum__qis__ry__body",
      "__quantum__qis__rz__body", "__quantum__qis__h__body"};

  std::vector<Instruction *> gatesToErase;
  FunctionCallee RZ = nullptr;
  FunctionCallee RY = nullptr;
  LLVMContext &rContext = module.getContext();
  IRBuilder<> builder(rContext);
  for (auto &function : module) {
    for (auto &block : function) {
      for (auto &instruction : block) {
        CallInst *callInstr = dyn_cast<CallInst>(&instruction);
        if (!callInstr)
          continue;
        Function *calledFunction = callInstr->getCalledFunction();
        for (std::string gateToDecompose : gatesToDecompose) {
          if (gateToDecompose == calledFunction->getName()) {
            int numberOfOperand = callInstr->getNumOperands();
            Value *theLastOperand = callInstr->getOperand(numberOfOperand - 2);
            Value *theAngle = callInstr->getOperand(0);
            LoadInst *loadofTheAngle = dyn_cast<LoadInst>(theAngle);
            ComplexMatrix theGate;
            if (!loadofTheAngle) {
              theGate = getTheMatrixOfGateFromInstructionName(gateToDecompose);
            } else {
              Value *theRotationAngle = loadofTheAngle->getPointerOperand();
              GlobalVariable *angleAsAConst =
                  dyn_cast_or_null<GlobalVariable>(theRotationAngle);
              ConstantFP *angleFP =
                  dyn_cast_or_null<ConstantFP>(angleAsAConst->getInitializer());
              double angle = angleFP->getValue().convertToDouble();
              theGate =
                  getTheMatrixOfGateFromInstructionName(gateToDecompose, angle);
            }
            if (!RZ) {
              Type *qubitType = theLastOperand->getType();
              FunctionType *rotationGateType = FunctionType::get(
                  Type::getVoidTy(rContext),
                  {Type::getDoubleTy(rContext), qubitType}, false);
              RZ = module.getOrInsertFunction(RZGate, rotationGateType);
              RY = module.getOrInsertFunction(RYGate, rotationGateType);
            }

            builder.SetInsertPoint((&instruction));
            std::vector<Value *> theAngles =
                getDecompositionAngles(rContext, theGate);
            builder.CreateCall(RZ, {theAngles[0], theLastOperand});
            builder.CreateCall(RY, {theAngles[1], theLastOperand});
            builder.CreateCall(RZ, {theAngles[2], theLastOperand});
            gatesToErase.push_back(&instruction);
          }
        }
      }
    }
  }

  for (Instruction *instr : gatesToErase) {
    instr->eraseFromParent();
  }
  return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirZExtTransformPass' as a
 * 'PassModule'.
 * @return QirZExtTransformPass
 */
extern "C" PassModule *loadQirPass() { return new QirZYZTransformPass(); }
