#include "../headers/QirMergeRotations.hpp"

using namespace llvm;

PreservedAnalyses QirMergeRotationsPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
        auto& Context = module.getContext();

        for (auto &function : module) {
            std::vector<CallInst*> gatesToRemove;
            std::vector<CallInst*> singletonContainer;
	        std::unordered_set<std::string> rotationGates = {"__quantum__qis__rx__body", "__quantum__qis__ry__body", "__quantum__qis__rz__body"};
            
	        for (auto &block : function) {
                for (auto &instruction : block) {
                    auto *current_instruction = dyn_cast<CallInst>(&instruction);

                    if (current_instruction) {
                        auto *current_function = current_instruction->getCalledFunction();
                    
                        if (current_function == nullptr)
                            continue;
                    
                        std::string current_name = static_cast<std::string>(current_function->getName());
                    
                        if (rotationGates.find(current_name) != rotationGates.end()) {
                            if (singletonContainer.size() == 0) {
                                singletonContainer.push_back(current_instruction);
                                continue;
                            }

                            CallInst *last_instruction = singletonContainer.back();
                            auto *last_function = last_instruction->getCalledFunction();

                            if (last_function->getName() != current_name) {
                                singletonContainer.clear();
                                singletonContainer.push_back(current_instruction);
                                continue;
                            }

                            Value *current_argument = current_instruction->getArgOperand(0);
                            Value *last_argument = last_instruction->getArgOperand(0);
                            auto *current_angle = dyn_cast_or_null<ConstantFP>(current_argument);
                            auto *last_angle = dyn_cast_or_null<ConstantFP>(last_argument);

                            if (!current_angle) {
                                if (LoadInst *ArgAsInstruction = dyn_cast_or_null<LoadInst>(current_argument)) {
                                    auto *rotationAngle = ArgAsInstruction->getPointerOperand();
                                    auto *angleAsAConst = dyn_cast_or_null<GlobalVariable>(rotationAngle);
                                    if (angleAsAConst) {
                                        current_angle = dyn_cast_or_null<ConstantFP>(angleAsAConst->getInitializer());
                                    }
                                }
                            }

                            if (!last_angle) {
                                if (LoadInst *ArgAsInstruction = dyn_cast_or_null<LoadInst>(last_argument)) {
                                    auto *rotationAngle = ArgAsInstruction->getPointerOperand();
                                    auto *angleAsAConst = dyn_cast_or_null<GlobalVariable>(rotationAngle);
                                    if (angleAsAConst) {
                                        last_angle = dyn_cast_or_null<ConstantFP>(angleAsAConst->getInitializer());
                                    }
                                }
                            }
                            
                            current_instruction->setArgOperand(0, ConstantFP::get(Context, (current_angle->getValue() + last_angle->getValue())));
                            gatesToRemove.push_back(last_instruction);
                            errs() << "\tRotation gates can be merged: " << current_name << '\n';
                        }
                        singletonContainer.clear();
                    }
                }
            }
            while (!gatesToRemove.empty()) {
                auto *gateToRemove = gatesToRemove.back();
                gateToRemove->eraseFromParent();
                gatesToRemove.pop_back();
            }
        }
    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirMergeRotationsPass();
}
