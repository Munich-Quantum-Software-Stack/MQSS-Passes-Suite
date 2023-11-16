/**
 * @file QirPauliGateAndHadamardSwitch.cpp
 * @brief Implementation of the 'QirPauliGateAndHadamardSwitchPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirPauliGateAndHadamardSwitch.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 */

#include "../headers/QirPauliGateAndHadamardSwitch.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses
QirPauliGateAndHadamardSwitchPass::run(Module &module,
                                       ModuleAnalysisManager & /*MAM*/) {
  auto &Context = module.getContext();
  std::unordered_set<std::string> pauliGates = {"__quantum__qis__x__body",
                                                "__quantum__qis__y__body",
                                                "__quantum__qis__z__body"};
  std::map<std::string, std::string> correspondingGates = {
      {"__quantum__qis__x__body", "__quantum__qis__z__body"},
      {"__quantum__qis__y__body", "__quantum__qis__y__body"},
      {"__quantum__qis__z__body", "__quantum__qis__x__body"},
  };

  for (auto &function : module) {
    std::vector<CallInst *> gatesToReplace;
    std::vector<CallInst *> currentGates;

    for (auto &block : function) {
      CallInst *prev_instruction = nullptr;

      for (auto &instruction : block) {
        auto *current_instruction = dyn_cast<CallInst>(&instruction);

        if (current_instruction) {
          auto *current_function = current_instruction->getCalledFunction();

          if (current_function == nullptr)
            continue;

          std::string current_name = current_function->getName().str();

          if (current_name == "__quantum__qis__h__body") {
            if (prev_instruction) {
              auto *prev_function =
                  dyn_cast<CallInst>(prev_instruction)->getCalledFunction();

              if (prev_function) {
                std::string previous_name = prev_function->getName().str();

                if (pauliGates.find(previous_name) != pauliGates.end()) {
                  currentGates.push_back(current_instruction);
                  gatesToReplace.push_back(prev_instruction);
                  errs() << "              Switching: " << previous_name
                         << " and " << current_name << '\n';
                }
              }
            }
          }
        }
        prev_instruction = current_instruction;
      }
    }
    while (!gatesToReplace.empty()) {
      auto *gateToReplace = gatesToReplace.back();
      auto *currentGate = currentGates.back();
      std::string gateName =
          gateToReplace->getCalledFunction()->getName().str();
      Function *newFunction = module.getFunction(correspondingGates[gateName]);
      if (!newFunction) {
        StructType *qubitType = StructType::getTypeByName(Context, "Qubit");
        PointerType *qubitPtrType = PointerType::getUnqual(qubitType);
        FunctionType *funcType =
            FunctionType::get(Type::getVoidTy(Context), {qubitPtrType}, false);
        newFunction = Function::Create(funcType, Function::ExternalLinkage,
                                       correspondingGates[gateName], module);
      }
      CallInst *newInst =
          CallInst::Create(newFunction, {gateToReplace->getOperand(0)});
      gateToReplace->eraseFromParent();
      newInst->insertAfter(currentGate);
      gatesToReplace.pop_back();
      currentGates.pop_back();
    }
  }
  return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirPauliGateAndHadamardSwitchPass'
 * as a 'PassModule'.
 * @return QirPauliGateAndHadamardSwitchPass
 */
extern "C" PassModule *loadQirPass() {
  return new QirPauliGateAndHadamardSwitchPass();
}
