#include "../headers/QirCommuteCnotRx.hpp"

using namespace llvm;

PreservedAnalyses QirCommuteCnotRxPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
    for (auto &function : module) {
        for (auto &block : function) {
            CallInst *prev_instruction = nullptr;

            for (auto &instruction : block) {
                auto *current_instruction = dyn_cast<CallInst>(&instruction);

                if (current_instruction) {
                    auto *current_function = current_instruction->getCalledFunction();
                
                    if (current_function == nullptr)
                        continue;
                
                    std::string current_name = current_function->getName().str();
                
                    if (current_name == "__quantum__qis__rx__body") {
                        if (prev_instruction) {
                            auto *prev_function = dyn_cast<CallInst>(prev_instruction)->getCalledFunction();
                            
                            if (prev_function) {
                                std::string previous_name = prev_function->getName().str();

                                if (previous_name == "__quantum__qis__cnot__body") {
                                    Value *previous_arg = prev_instruction->getArgOperand(1);
                                    Value *current_arg  = current_instruction->getArgOperand(1);
                                    
                                    if (previous_arg == current_arg) {
                                        current_instruction->moveBefore(prev_instruction);
                                        errs() << "\tCommuting: " << previous_name << " and " << current_name << '\n';
                                    }
                                }
                            }
                        }
                    }
                }
                prev_instruction = current_instruction;
            }
        }
    }
    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirCommuteCnotRxPass();
}
