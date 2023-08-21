#include "../headers/QirBarrierBeforeFinalMeasurements.h"

using namespace llvm;

PreservedAnalyses QirBarrierBeforeFinalMeasurementsPass::run(Module *module, ModuleAnalysisManager &/*mam*/) {
    std::vector<Instruction*> mz_instructions;
    bool barrier_found = false;
    for(auto &function : *module){
    	for(auto &block : function){
            for(auto &instruction : block){
            	CallInst *call_instruction = dyn_cast<CallInst>(&instruction);

            	if(call_instruction){
                    Function *mz_function = call_instruction->getCalledFunction();

                    if(mz_function == nullptr)
                    	continue;

                    std::string call_name = static_cast<std::string>(mz_function->getName());

                    if(call_name == "__quantum__qis__mz__body")
                    	mz_instructions.push_back(&instruction);
		            else if(call_name == "__quantum__qis__barrier__body")
                    	barrier_found = true;
            	}
            }
	    }

    	LLVMContext &Ctx = function.getContext();

    	FunctionType* function_type = FunctionType::get(
            Type::getVoidTy(Ctx),       // return void
            false);                     // no variable arguments

        if(mz_instructions.empty())
            return PreservedAnalyses::none();

        Function *barrier_function;
        if(barrier_found)
            barrier_function = module->getFunction("__quantum__qis__barrier__body");
        else{
            barrier_function = Function::Create(
                function_type,
                function.getLinkage(), //Function::ExternalWeakLinkage,
                "__quantum__qis__barrier__body",
                module);
        }

        while(!mz_instructions.empty()){
            Instruction *mz_instruction = mz_instructions.back();

            CallInst::Create(
                function_type,
                barrier_function,       // new function
                "",                     // no name required
                mz_instruction);        // insert before

            mz_instructions.pop_back();
        }
    }

    return PreservedAnalyses::all();
}

extern "C" PassModule* createQirBarrierBeforeFinalMeasurementsPass() {
    return new QirBarrierBeforeFinalMeasurementsPass();
}
