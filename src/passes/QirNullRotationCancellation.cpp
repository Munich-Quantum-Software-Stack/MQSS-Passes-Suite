#include "../headers/QirNullRotationCancellation.hpp"

using namespace llvm;

PreservedAnalyses QirNullRotationCancellationPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
    auto& Context = module.getContext();

    const double zeroValue   = 0.0;
    const double piValue     = 3.14159265;
    ConstantFP* zeroConstant = ConstantFP::get(Context, APFloat(zeroValue));
    ConstantFP* piConstant   = ConstantFP::get(Context, APFloat(piValue));

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
			Value *argument = current_function->getArg(0);
			errs() << "DEBUG " << current_name << '\n';
			if (argument->getType()->isDoubleTy()) {
                            errs() << "DEBUG first check" << *argument->getType() << '\n';
			    if (ConstantFP *constFP = dyn_cast<ConstantFP>(argument)) {
                                errs() << "DEBUG into casting" << '\n';
				double rotation = constFP->getValueAPF().convertToDouble();
                                errs() << "DEBUG getting rotation value: " << rotation << '\n';
                                if (rotation == zeroValue || rotation == 2*piValue) {
                                    rotationGatesToRemove.push_back(current_instruction);
                                    errs() << "\tRedundant rotation found: " << current_name << '\n';
                                }
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
