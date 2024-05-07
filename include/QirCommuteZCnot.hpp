/**
 * @file QirCommuteZCnot.hpp
 * @brief Declaration of the 'QirCommuteZCnotPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirCommuteZCnotPass
 * @brief This pass swaps adjacent Z and CNOT gates whenever found in this
 * specific order.
 */
class QirCommuteZCnotPass : public AgnosticPassModule
{
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

} // namespace llvm
