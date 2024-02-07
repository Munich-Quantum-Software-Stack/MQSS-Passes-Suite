/**
 * @file QirFunctionAnnotator.hpp
 * @brief Declaration of the 'QirFunctionAnnotatorPass' class.
 */

#pragma once

#include <PassModule.hpp>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm
{

/**
 * @class QirFunctionAnnotatorPass
 * @brief This pass edits the attributes of those gates to
 * be replaced according to information taken from the metadata.
 */
class QirFunctionAnnotatorPass : public PassModule
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
