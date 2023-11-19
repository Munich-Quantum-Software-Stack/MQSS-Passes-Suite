/**
 * @file QirNullRotationCancellation.cpp
 * @brief Implementation of the 'QirNullRotationCancellationPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirNullRotationCancellation.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass removes rotation gates with null rotation, that is rotation by 0 or
 * by 2pi multiplies.
 */

#include "../headers/QirNullRotationCancellation.hpp"

using namespace llvm;

/**
 * Checks if provided angle is a 2p multiple.
 * @param double The rotation angle.
 * @return bool
 */
bool checkDoublePiMultiplies(double angle) {
  const double doublePi = 2 * 3.14159265358979323846;
  if (std::fmod(angle, doublePi) == 0)
    return true;
  return false;
}

/**
 * Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses
QirNullRotationCancellationPass::run(Module &module,
                                     ModuleAnalysisManager & /*MAM*/) {
  std::unordered_set<std::string> rotationGates = {"__quantum__qis__rx__body",
                                                   "__quantum__qis__ry__body",
                                                   "__quantum__qis__rz__body"};

  for (auto &function : module) {
    std::vector<CallInst *> rotationGatesToRemove;

    for (auto &block : function) {
      for (auto &instruction : block) {
        auto *current_instruction = dyn_cast<CallInst>(&instruction);

        if (!current_instruction)
          continue;

        auto *current_function = current_instruction->getCalledFunction();

        if (current_function == nullptr)
          continue;

        std::string current_name = current_function->getName().str();

        if (rotationGates.find(current_name) != rotationGates.end()) {
          Value *arg = current_instruction->getArgOperand(0);
          auto *angleFP = dyn_cast_or_null<ConstantFP>(arg);

          if (!angleFP) {
            if (LoadInst *argAsInstruction = dyn_cast_or_null<LoadInst>(arg)) {
              auto *rotationAngle = argAsInstruction->getPointerOperand();
              auto *angleAsAConst =
                  dyn_cast_or_null<GlobalVariable>(rotationAngle);
              if (angleAsAConst)
                angleFP = dyn_cast_or_null<ConstantFP>(
                    angleAsAConst->getInitializer());
            }
          }

          if (angleFP) {
            auto angle = angleFP->getValue().convertToDouble();

            if (angleFP->isZero() || checkDoublePiMultiplies(angle)) {
              rotationGatesToRemove.push_back(current_instruction);
              errs() << "[Pass].............Redundant rotation found: "
                     << current_name << '\n';
            }
          }
        }
      }
    }

    while (!rotationGatesToRemove.empty()) {
      auto *rotationGateToRemove = rotationGatesToRemove.back();
      rotationGateToRemove->eraseFromParent();
      rotationGatesToRemove.pop_back();
    }
  }

  return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirNullRotationCancellationPass' as
 * a 'PassModule'.
 * @return QirNullRotationCancellationPass
 */
extern "C" PassModule *loadQirPass() {
  return new QirNullRotationCancellationPass();
}
