/**
 * @file QirCommuteRxCnot.hpp
 * @brief Declaration of the 'QirCommuteRxCnotPass' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirCommuteRxCnotPass
 * @brief This pass swaps adjacent Rx and CNOT gates whenever found in this specific order. 
 */
class QirCommuteRxCnotPass : public PassModule {
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

