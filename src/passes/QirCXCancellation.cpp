#include "../headers/QirCXCancellation.h"

using namespace llvm;

PreservedAnalyses QirCXCancellationPass::run(Module *module, ModuleAnalysisManager &/*mam*/) {
    std::vector<CallInst*> cx_instructions;
    std::vector<CallInst*> single_cx;
    for(auto &function : *module){
	for(auto &block : function){
            for(auto &instruction : block){
                auto *current_instruction = dyn_cast<CallInst>(&instruction);

                if(current_instruction){
                    auto *current_function = current_instruction->getCalledFunction();
                
                    if(current_function == nullptr)
                        continue;
                
                    std::string current_name = static_cast<std::string>(current_function->getName());
                
                    if(current_name == "__quantum__qis__cnot__body"){
			if(single_cx.size() == 0){
			    single_cx.push_back(current_instruction);
			    continue;
			}

			CallInst *last_instruction = single_cx.back();
					
			cx_instructions.push_back(last_instruction);
			cx_instructions.push_back(current_instruction);
		    }
		    single_cx.clear();
                }
            }
        }

	assert(((void)"Programming error: please report this issue", cx_instructions.size() % 2 == 0));

        while(!cx_instructions.empty()){
            auto *cx_instruction = cx_instructions.back();
	    cx_instruction->eraseFromParent();
            cx_instructions.pop_back();
        }
    }
    return PreservedAnalyses::all();
}

extern "C" PassModule* createQirPass() {
    return new QirCXCancellationPass();
}
