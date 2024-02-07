/**
 * @file QirXYXDecomposition.hpp
 * @brief Declaration of the 'QirXYXDecomposition' class.
 */
#pragma once

#include "utilities.hpp"
#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirXYXDecompositionPass
 * @brief Xhis Pass creates a XYX Decomposition of given gates
 * (gatesToDecompose).
 */
class QirXYXDecompositionPass : public PassModule
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

    std::vector<Value *> getDecompositionAngles(LLVMContext &context,
                                                ComplexMatrix theGate);
};

} // namespace llvm
