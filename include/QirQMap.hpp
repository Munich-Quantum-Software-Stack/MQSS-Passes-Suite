/**
 * @file QirQMap.hpp
 * @brief Declaration of the 'QirQMapPass' class.
 */

#pragma once

#include <PassModule.hpp>

#include "ArchitectureFactory.hpp"
#include "QuantumComputation.hpp"
#include "heuristic/HeuristicMapper.hpp"

#include <qdmi.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm
{

/**
 * @class QirQMapPass
 * @brief TODO
 */
class QirQMapPass : public SpecificPassModule
{
  public:

    /**
     * @brief Applies this pass to the QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @param dev The QDMI Device.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM,
                          QDMI_Device dev);
};

} // namespace llvm
