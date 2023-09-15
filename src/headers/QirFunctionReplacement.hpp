#pragma once

#include "PassModule.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

struct FunctionRegister {
    using FunctionMap    = std::unordered_map<std::string, Function*>;
    using ReplacementMap = std::unordered_map<Function*, Function*>;
    using CallList       = std::vector<CallInst*>;

    FunctionMap    name_to_function_pointer{};
    ReplacementMap functions_to_replace{};
    CallList       calls_to_replace{};
};

class QirFunctionReplacementPass : public PassModule {
public:
	using Result = FunctionRegister;

    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
	Result runFunctionReplacementAnalysis(Module &module);
};

}

