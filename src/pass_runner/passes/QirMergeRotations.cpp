/**
 * @file QirMergeRotations.cpp
 * @brief Implementation of the 'QirMergeRotationsPass' class. <a href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirMergeRotations.cpp?ref_type=heads">Go to the source code of this file.</a>
 *
 * This pass merges equivalent rotation gates into single rotation. 
 */

#include "../headers/QirMergeRotations.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
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
                            errs() << "[Pass].............Rotation gates can be merged: " << gateTypeToMerge << '\n';
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

/**
 * @brief External function for loading the 'QirMergeRotationsPass' as a 'PassModule'.
 * @return QirMergeRotationsPass
 */
extern "C" PassModule* loadQirPass() {
    return new QirMergeRotationsPass();
}
