/**
 * @file QirBarrierBeforeFinalMeasurements.hpp
 * @brief Declaration of the 'QirBarrierBeforeFinalMeasurementsPass' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm
{

/**
 * @class QirBarrierBeforeFinalMeasurementsPass
 * @brief This pass inserts a barrier before each measurement.
 */
class QirBarrierBeforeFinalMeasurementsPass : public PassModule
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
