/**
 * @file QirFunctionReplacement.hpp
 * @brief Declaration of the 'QirFunctionReplacementPass' class.
 */

#pragma once

#include "PassModule.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

/**
 * @struct FunctionRegister
 * @brief TODO
 */
struct FunctionRegister {
    using FunctionMap    = std::unordered_map<std::string, Function*>;
    using ReplacementMap = std::unordered_map<Function*, Function*>;
    using CallList       = std::vector<CallInst*>;

    FunctionMap    name_to_function_pointer{};  /**< TODO. */
    ReplacementMap functions_to_replace{};      /**< TODO. */
    CallList       calls_to_replace{};          /**< TODO. */
};

/**
 * @class QirFunctionReplacementPass
 * @brief This pass replaces gates with functions describing their decompositions.
 */
class QirFunctionReplacementPass : public PassModule {
public:
	using Result = FunctionRegister;

    /**
     * @brief Applies this pass to the QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
    
    /**
     * @brief Applies a function replacement analysis pass to the 
     * QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @return Result
     */
	Result runFunctionReplacementAnalysis(Module &module);
};

}

