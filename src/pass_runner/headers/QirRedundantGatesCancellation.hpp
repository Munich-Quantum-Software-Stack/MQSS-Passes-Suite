/**
 * @file QirRedundantGatesCancellation.hpp
 * @brief Declaration of the 'QirRedundantGatesCancellationPass' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirRedundantGatesCancellationPass
 * @brief This pass removes redundant one-qubit gates, that is, 
 * equivalent gates acting back to back on the same qubit.
 */
class QirRedundantGatesCancellationPass : public PassModule {
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

