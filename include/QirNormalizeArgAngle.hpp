/**
 * @file QirNormalizeArgAngle.hpp
 * @brief Declaration of the 'QirNormalizeArgAnglePass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirNormalizeArgAnglePass
 * @brief This pass normalizes the angle of rotation gates within [0, 2*Pi).
 */
class QirNormalizeArgAnglePass : public PassModule
{
  public:
    /**
     * @brief Applies this pass to the QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM,
                          QDMI_Device dev);
};

} // namespace llvm
