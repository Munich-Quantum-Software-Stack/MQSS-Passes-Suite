/**
 * @file QirHZHToX.hpp
 * @brief Declaration of the 'QirHZHToX' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm
{

/**
 * @class QirHZHToX
 * @brief This pass changes adjacent H, Z and H gates into X whenever found in
 * this order.
 */
class QirHZHToXPass : public AgnosticPassModule
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
