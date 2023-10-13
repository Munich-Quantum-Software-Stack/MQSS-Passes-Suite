#include "../headers/QirRedundantGatesCancellation.hpp"

using namespace llvm;

PreservedAnalyses QirRedundantGatesCancellationPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
    QirMetadata &qirMetadata = QirPassRunner::getInstance().getMetadata();

    for (auto reversibleGate : qirMetadata.reversibleGates) {
        for (auto &function : module) {
            std::vector<CallInst*> gatesToRemove;
            std::vector<CallInst*> singletonContainer;
            for (auto &block : function) {
                for (auto &instruction : block) {
                    auto *current_instruction = dyn_cast<CallInst>(&instruction);

                    if (current_instruction) {
                        auto *current_function = current_instruction->getCalledFunction();
                    
                        if (current_function == nullptr)
                            continue;
                    
                        std::string current_name = static_cast<std::string>(current_function->getName());
                    
                        if (current_name == reversibleGate) {
                            if (singletonContainer.size() == 0) {
                                singletonContainer.push_back(current_instruction);
                                continue;
                            }

                            CallInst *last_instruction = singletonContainer.back();
                                    
                            gatesToRemove.push_back(last_instruction);
                            gatesToRemove.push_back(current_instruction);
                            
                            errs() << "              Redundant gate pair found: " << reversibleGate << '\n';
                        }
                        singletonContainer.clear();
                    }
                }
            }

            assert(((void)"Number of gates to be removed is not even", gatesToRemove.size() % 2 == 0));

            while (!gatesToRemove.empty()) {
                auto *gateToRemove = gatesToRemove.back();
                gateToRemove->eraseFromParent();
                gatesToRemove.pop_back();
            }
        }
    }
    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirRedundantGatesCancellationPass();
}
