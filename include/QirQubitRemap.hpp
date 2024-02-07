/**
 * @file QirQubitRemap.hpp
 * @brief Declaration of the 'QirQubitRemapPass' class.
 */

#pragma once

#include <PassModule.hpp>

#include <functional>
#include <unordered_map>
#include <vector>

namespace llvm
{

/**
 * @class QirQubitRemapPass
 * @brief This pass remaps qubits to independent values.
 */
class QirQubitRemapPass : public PassModule
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
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM,
                          QDMI_Device dev);
};

} // namespace llvm
