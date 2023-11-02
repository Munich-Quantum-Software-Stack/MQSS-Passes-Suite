/**
 * @file QirCommuteXCnot.hpp
 * @brief Declaration of the 'QirCommuteXCnotPass' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirCommuteXCnotPass
 * @brief This pass swaps adjacent X and CNOT gates whenever found in this specific order. 
 */
class QirCommuteXCnotPass : public PassModule {
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

