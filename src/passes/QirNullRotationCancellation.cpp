#include "../headers/QirNullRotationCancellation.hpp"

using namespace llvm;

PreservedAnalyses QirNullRotationCancellationPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
    auto& Context = module.getContext();

    const double zeroValue = 0.0;
    const double twoPiValue = 2 * 3.14159265;
    ConstantFP* zeroConstant = ConstantFP::get(Context, APFloat(zeroValue));
    ConstantFP* twoPiConstant = ConstantFP::get(Context, APFloat(twoPiValue));

    std::unordered_set<std::string> rotationGates = {"__quantum__qis__rx__body", "__quantum__qis__ry__body", "__quantum__qis__rz__body"};
    for (auto &function : module) {
        std::vector<CallInst*> rotationGatesToRemove;
        for (auto &block : function) {
            for (auto &instruction : block) {
                auto *current_instruction = dyn_cast<CallInst>(&instruction);
		
		if (current_instruction) {
                    auto *current_function = current_instruction->getCalledFunction();
                    
                    if (current_function == nullptr)
                        continue;
                    
                    std::string current_name = current_function->getName().str();

	            if (rotationGates.find(current_name) != rotationGates.end()) {
		        Value* arg = current_instruction->getArgOperand(0);

			    // Case in which double %zero is an argument
			    if (LoadInst* ArgsAsInstruction = dyn_cast_or_null<LoadInst>(arg)) {
			        Value* rotationAngle = ArgsAsInstruction->getPointerOperand();
				if (GlobalVariable* angleAsAConst = dyn_cast_or_null<GlobalVariable>(rotationAngle)) {
				    if (ConstantFP* angleFP = dyn_cast_or_null<ConstantFP>(angleAsAConst->getInitializer())) {
				        if (angleFP->isZero() || angleFP == twoPiConstant) {
                                    	    rotationGatesToRemove.push_back(current_instruction);
                                    	    errs() << "\tRedundant rotation found: " << current_name << '\n';
					}
				    }
				}
			    // Case in which double 0.0 is an argument
			    } else if (ConstantFP* angleFP = dyn_cast_or_null<ConstantFP>(arg)) {
                                if (angleFP->isZero() || angleFP == twoPiConstant) {
                                    rotationGatesToRemove.push_back(current_instruction);
                                    errs() << "\tRedundant rotation found: " << current_name << '\n';
				}
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

extern "C" PassModule* createQirPass() {
    return new QirNullRotationCancellationPass();
}
