/**
 * @file QirZYZDecomposition.hpp
 * @brief Declaration of the 'QirZYZDecomposition' class.
 */
#pragma once

#include "utilities.hpp"
#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirZYZDecompositionPass
 * @brief This Pass creates a ZYZ Decomposition of given gates.
 */
class QirZYZDecompositionPass : public AgnosticPassModule
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

    std::vector<Value *> getDecompositionAngles(LLVMContext &context,
                                                ComplexMatrix theGate);
    std::vector<double> getDecompositionAnglesAsNumber(LLVMContext &context,
                                                       ComplexMatrix theGate);
};

} // namespace llvm
