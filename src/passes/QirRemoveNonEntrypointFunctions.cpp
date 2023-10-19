#include "../headers/QirRemoveNonEntrypointFunctions.hpp"

using namespace llvm;

PreservedAnalyses QirRemoveNonEntrypointFunctionsPass::run(Module &module, ModuleAnalysisManager &MAM) {
    std::vector<Function*> functions_to_delete;

    for(auto &function : module)
        if(!function.isDeclaration() && !function.hasFnAttribute("entry_point"))
            functions_to_delete.push_back(&function);

    for(auto &function : functions_to_delete)
        if(function->use_empty())
            function->eraseFromParent();

    return PreservedAnalyses::none();
}

extern "C" PassModule* loadQirPass() {
    return new QirRemoveNonEntrypointFunctionsPass();
}

