#include "../headers/QirFunctionReplacement.hpp"

using namespace llvm;

QirFunctionReplacementPass::Result QirFunctionReplacementPass::runFunctionReplacementAnalysis(Module &module) {
    FunctionRegister ret;
    
    // Registering all functions
    for (auto &function : module)
        ret.name_to_function_pointer[static_cast<std::string>(function.getName())] = &function;

    // Registering replacements
    for (auto &function : module) {
        if (function.hasFnAttribute("replaceWith")) {
            auto attr = function.getFnAttribute("replaceWith");
            errs() << "              Function has 'replaceWith' attribute: " << static_cast<std::string>(function.getName()) << '\n';

            if (!attr.isStringAttribute()) {
                errs() << "              Warning: Expected string attribute for attribute 'replaceWith'\n";
                continue;
            }

            auto name = static_cast<std::string>(attr.getValueAsString());
            auto it   = ret.name_to_function_pointer.find(name);

            errs() << "              Function is a replacement           : " << name << '\n';

            // Ignoring replacements that were not found
            if (it == ret.name_to_function_pointer.end()) {
                errs() << "              Warning: replacement not found\n";
                continue;
            }

            // Checking function signature
            std::string signature1;
            raw_string_ostream ostream1(signature1);
            ostream1 << *function.getFunctionType();

            std::string signature2;
            raw_string_ostream ostream2(signature2);
            ostream2 << *it->second->getFunctionType();

            if (signature1 != signature2) {
                errs() << "              Warning: Expected string attribute for attribute 'replaceWith'\n";
                continue;
            }

            // Registering replacement
            ret.functions_to_replace[&function] = it->second;
        }
    }

    for (auto &function : module)
        for (auto &block : function)
            for (auto &instr : block) {
                auto call_instr = dyn_cast<CallInst>(&instr);
                if (call_instr == nullptr)
                    continue;

                auto function_ptr = call_instr->getCalledFunction();
                auto it           = ret.functions_to_replace.find(function_ptr);

                if (function_ptr == nullptr || it == ret.functions_to_replace.end())
                    continue;

                ret.calls_to_replace.push_back(call_instr);
            }

    return ret;
}

PreservedAnalyses QirFunctionReplacementPass::run(Module &module, ModuleAnalysisManager &MAM) {
    IRBuilder<> builder(module.getContext());
    auto result = runFunctionReplacementAnalysis(module);

    for (auto& call_instr : result.calls_to_replace) {
        auto function = call_instr->getCalledFunction();
        auto it       = result.functions_to_replace.find(function);

        if (function == nullptr || it == result.functions_to_replace.end())
            continue;

        std::vector<Value*> arguments;
        for (std::size_t i = 0; i < call_instr->arg_size(); ++i)
            arguments.emplace_back(call_instr->getArgOperand(i));

        builder.SetInsertPoint(dyn_cast<Instruction>(call_instr));
        auto new_call = builder.CreateCall(it->second, arguments);
        new_call->takeName(call_instr);
        call_instr->replaceAllUsesWith(new_call);
        call_instr->eraseFromParent();
    }

	return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirFunctionReplacementPass();
}
