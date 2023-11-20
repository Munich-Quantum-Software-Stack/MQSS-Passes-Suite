/**
 * @file QirHadamardAndZGateSwitch.cpp
<<<<<<< HEAD
 * @brief Implementation of the 'QirHadamardAndZGateSwitchPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirHadamardAndZGateSwitch.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
=======
 * @brief Implementation of the 'QirHadamardAndZGateSwitchPass' class. <a href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirHadamardAndZGateSwitch.cpp?ref_type=heads">Go to the source code of this file.</a>
>>>>>>> 5356c34 (Resolving conflicts against NoSockets branch)
 *
 */

#include "../headers/QirHadamardAndZGateSwitch.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
<<<<<<< HEAD
PreservedAnalyses
QirHadamardAndZGateSwitchPass::run(Module &module,
                                   ModuleAnalysisManager & /*MAM*/) {
  auto &Context = module.getContext();

  for (auto &function : module) {
    std::vector<CallInst *> gatesToReplace;
    std::vector<CallInst *> previousGates;

    for (auto &block : function) {
      CallInst *prev_instruction = nullptr;

      for (auto &instruction : block) {
        auto *current_instruction = dyn_cast<CallInst>(&instruction);

        if (current_instruction) {
          auto *current_function = current_instruction->getCalledFunction();

          if (current_function == nullptr)
            continue;

          std::string current_name = current_function->getName().str();

          if (current_name == "__quantum__qis__z__body") {
            if (prev_instruction) {
              auto *prev_function =
                  dyn_cast<CallInst>(prev_instruction)->getCalledFunction();

              if (prev_function) {
                std::string previous_name = prev_function->getName().str();

                if (previous_name == "__quantum__qis__h__body") {
                  previousGates.push_back(prev_instruction);
                  gatesToReplace.push_back(current_instruction);
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
      auto *prevGate = previousGates.back();
      std::string gateName =
          gateToReplace->getCalledFunction()->getName().str();
      Function *newFunction = module.getFunction("__quantum__qis__x__body");
      if (!newFunction) {
        StructType *qubitType = StructType::getTypeByName(Context, "Qubit");
        PointerType *qubitPtrType = PointerType::getUnqual(qubitType);
        FunctionType *funcType =
            FunctionType::get(Type::getVoidTy(Context), {qubitPtrType}, false);
        newFunction = Function::Create(funcType, Function::ExternalLinkage,
                                       "__quantum__qis__x__body", module);
      }
      CallInst *newInst =
          CallInst::Create(newFunction, {gateToReplace->getOperand(0)});
      gateToReplace->eraseFromParent();
      newInst->insertBefore(prevGate);
      gatesToReplace.pop_back();
      previousGates.pop_back();
    }
  }
  return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirHadamardAndZGateSwitchPass' as a
 * 'PassModule'.
 * @return QirHadamardAndZGateSwitchPass
 */
extern "C" PassModule *loadQirPass() {
  return new QirHadamardAndZGateSwitchPass();
=======
PreservedAnalyses QirHadamardAndZGateSwitchPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
    auto& Context = module.getContext();

    for (auto &function : module) {
        std::vector<CallInst*> gatesToReplace;
        std::vector<CallInst*> previousGates;

        for (auto &block : function) {
            CallInst *prev_instruction = nullptr;

            for (auto &instruction : block) {
                auto *current_instruction = dyn_cast<CallInst>(&instruction);

                if (current_instruction) {
                    auto *current_function = current_instruction->getCalledFunction();
                
                    if (current_function == nullptr)
                        continue;
                
                    std::string current_name = current_function->getName().str();
                
                    if (current_name == "__quantum__qis__z__body") {
                        if (prev_instruction) {
                            auto *prev_function = dyn_cast<CallInst>(prev_instruction)->getCalledFunction();
                            
                            if (prev_function) {
                                std::string previous_name = prev_function->getName().str();

                                if (previous_name == "__quantum__qis__h__body") {
                                    previousGates.push_back(prev_instruction);
                                    gatesToReplace.push_back(current_instruction);
                                    errs() << "              Switching: " << previous_name << " and " << current_name << '\n';
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
            auto *prevGate = previousGates.back();
            std::string gateName = gateToReplace->getCalledFunction()->getName().str();
            Function *newFunction = module.getFunction("__quantum__qis__x__body");
            if (!newFunction) {
                StructType *qubitType = StructType::getTypeByName(Context, "Qubit");
                PointerType *qubitPtrType = PointerType::getUnqual(qubitType);
                FunctionType *funcType = FunctionType::get(
                    Type::getVoidTy(Context),
                    {
                        qubitPtrType
                    },
                    false
                );
                newFunction = Function::Create(
                    funcType,
                    Function::ExternalLinkage,
                    "__quantum__qis__x__body",
                    module
                );
            }
            CallInst *newInst = CallInst::Create(newFunction, {gateToReplace->getOperand(0)});
            gateToReplace->eraseFromParent();
            newInst->insertBefore(prevGate);
            gatesToReplace.pop_back();
            previousGates.pop_back();
        }
    }
    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirHadamardAndZGateSwitchPass' as a 'PassModule'.
 * @return QirHadamardAndZGateSwitchPass
 */
extern "C" PassModule* loadQirPass() {
    return new QirHadamardAndZGateSwitchPass();
>>>>>>> 5356c34 (Resolving conflicts against NoSockets branch)
}
