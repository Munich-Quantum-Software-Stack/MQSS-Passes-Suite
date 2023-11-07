/**
 * @file QirRemoveNonEntrypointFunctions.hpp
 * @brief Declaration of the 'QirRemoveNonEntrypointFunctionsPass' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirRemoveNonEntrypointFunctionsPass
 * @brief This pass removes ALL non-entry functions.
 */
class QirRemoveNonEntrypointFunctionsPass : public PassModule {
public:
    /**
     * @brief Applies this pass to the QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

