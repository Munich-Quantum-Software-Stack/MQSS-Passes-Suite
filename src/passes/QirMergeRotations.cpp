#include "../headers/QirMergeRotations.hpp"

using namespace llvm;

PreservedAnalyses QirMergeRotationsPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
        auto& Context = module.getContext();

        for (auto &function : module) {
            std::vector<CallInst*> gatesToRemove;
            std::vector<CallInst*> gatesToMerge;
            std::string gateTypeToMerge;
	        std::unordered_set<std::string> rotationGates = {"__quantum__qis__rx__body", "__quantum__qis__ry__body", "__quantum__qis__rz__body"};
            
	        for (auto &block : function) {
                for (auto &instruction : block) {
                    auto *currentInstruction = dyn_cast<CallInst>(&instruction);

                    if (currentInstruction) {
                        auto *currentFunction = currentInstruction->getCalledFunction();
                    
                        if (currentFunction == nullptr)
                            continue;
                    
                        std::string currentName = static_cast<std::string>(currentFunction->getName());

                        if (currentName == gateTypeToMerge) {
                            gatesToMerge.push_back(currentInstruction);
                            continue;
                        }

                        if (gatesToMerge.size() > 1) {
                            double sumRotations = 0;
                            for (auto &instruction : gatesToMerge) { 
                                Value *argument = instruction->getArgOperand(0);
                                auto *angle = dyn_cast_or_null<ConstantFP>(argument);

                                if (!angle) {
                                    if (LoadInst *argAsInstruction = dyn_cast_or_null<LoadInst>(argument)) {
                                        auto *rotationAngle = argAsInstruction->getPointerOperand();
                                        auto *angleAsAConst = dyn_cast_or_null<GlobalVariable>(rotationAngle);
                                        if (angleAsAConst) {
                                            angle = dyn_cast_or_null<ConstantFP>(angleAsAConst->getInitializer());
                                        }
                                    }
                                }

                                sumRotations += angle->getValue().convertToDouble();
                                
                                if (instruction == gatesToMerge.back()) {
                                    instruction->setArgOperand(0, ConstantFP::get(Context, APFloat(sumRotations)));
                                } else {
                                    gatesToRemove.push_back(instruction);
                                }
                            }
                            errs() << "\tRotation gates can be merged: " << gateTypeToMerge << '\n';
                        }

                        gatesToMerge.clear();
                        gateTypeToMerge = "";
                        
                        if (rotationGates.find(currentName) != rotationGates.end()) {
                            gatesToMerge.push_back(currentInstruction);
                            gateTypeToMerge = currentName;
                            continue;
                        }

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
