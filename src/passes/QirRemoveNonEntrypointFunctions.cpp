#include "../headers/QirRemoveNonEntrypointFunctions.h"

using namespace llvm;

PreservedAnalyses QirRemoveNonEntrypointFunctionsPass::run(Module *module, ModuleAnalysisManager &/*mam*/){
    std::vector<Function*> functions_to_delete;

    for(auto &function : *module)
        if(!function.isDeclaration() && !function.hasFnAttribute("entry_point"))
            functions_to_delete.push_back(&function);

    for(auto &function : functions_to_delete)
        if(function->use_empty())
            function->eraseFromParent();

    return PreservedAnalyses::all();
}

extern "C" PassModule* createQirRemoveNonEntrypointFunctionsPass() {
    return new QirRemoveNonEntrypointFunctionsPass();
}

