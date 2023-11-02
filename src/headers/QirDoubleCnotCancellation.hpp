/**
 * @file QirDoubleCnotCancellation.hpp
 * @brief Declaration of the 'QirDoubleCnotCancellationPass' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirDoubleCnotCancellationPass
 * @brief This pass removes two sequential Cnots acting on the same qubit.
 */
class QirDoubleCnotCancellationPass : public PassModule {
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

