/**
 * @file QirResourceAnnotation.hpp
 * @brief Declaration of the 'QirResourceAnnotationPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirResourceAnnotationPass
 * @brief This pass edits the attributes of the entry function
 * with the appropriate number of qubits and classical bits after
 * invoking the QirAllocationAnalysis pass.
 */
class QirResourceAnnotationPass : public AgnosticPassModule
{
  public:
    /**
     * @enum ResourceType
     * @brief Enumerated type for representing types of resources
     */
    enum ResourceType
    {
        None,  /**< Neither Qubit nor Result type. */
        Qubit, /**< Qubit type. */
        Result /**< Result type. */
    };

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
