/**
 * @file QirNormalizeArgAngle.cpp
 * @brief Implementation of the 'QirNormalizeArgAnglePass' class. <a href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirNormalizeArgAngle.cpp?ref_type=heads">Go to the source code of this file.</a>
 *
 * Adapted from:
 */
#include "../headers/QirNormalizeArgAngle.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirNormalizeArgAnglePass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
    auto& Context = module.getContext();

    for (auto &function : module) {
        for (auto &block : function) {
            for (auto &instruction : block) {
                auto *call_instruction = dyn_cast<CallInst>(&instruction);

                if (!call_instruction)
                    continue;

                auto *call_function = call_instruction->getCalledFunction();
            
                if (call_function == nullptr)
                    continue;
                
                std::string call_name = call_function->getName().str();
            
                if (call_name == "__quantum__qis__rx__body" ||
                    call_name == "__quantum__qis__ry__body" ||
                    call_name == "__quantum__qis__rz__body") {
                    
					Value *argument = call_instruction->getArgOperand(0);
                    auto *angleFP = dyn_cast_or_null<ConstantFP>(argument);

                    if (!angleFP) {
                        if (LoadInst *ArgsAsInstruction = dyn_cast_or_null<LoadInst>(argument)) {
                            auto *rotationAngle = ArgsAsInstruction->getPointerOperand();
                            auto *angleAsAConst = dyn_cast_or_null<GlobalVariable>(rotationAngle);
                            
                            if (angleAsAConst)
                                angleFP = dyn_cast_or_null<ConstantFP>(angleAsAConst->getInitializer());
                        }
                    }

                    if (angleFP) {
                        double argValue = angleFP->getValueAPF().convertToDouble();

                        // Calculate the greatest multiple of 2*Pi smaller than the argument
                        double pi = 3.14159265358979323846;
                        double multipleOf2Pi = floor(argValue / (2.0 * pi)) * (2.0 * pi);

                        // Calculate the result
                        double result = argValue - multipleOf2Pi;

                        // Replace the argument with the result
                        call_instruction->setArgOperand(0, ConstantFP::get(Context, APFloat(result)));
					} 
                }
            }
        }
    }

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirNormalizeArgAnglePass' as a 'PassModule'.
 * @return QirNormalizeArgAnglePass
 */
extern "C" PassModule* loadQirPass() {
    return new QirNormalizeArgAnglePass();
}

