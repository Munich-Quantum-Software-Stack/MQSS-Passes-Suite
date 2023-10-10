#pragma once

#include "llvm.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

struct FunctionValidation {
    std::vector<PoisonValue*> poisoned_instructions;
    std::vector<UndefValue*>  undefined_instructions;

    bool qubits_present;
    bool results_present;
};

class QirFunctionValidationPass : public AnalysisInfoMixin<QirFunctionValidationPass> {
public:
    using Result = FunctionValidation;

    Result ValidationResult;

    PreservedAnalyses run(Function &function, FunctionAnalysisManager &FAM);
};

}

