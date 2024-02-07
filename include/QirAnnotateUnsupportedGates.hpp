/**
 * @file QirAnnotateUnsupportedGates.hpp
 * @brief Declaration of the 'QirAnnotateUnsupportedGatesPass' class.
 */

#pragma once

#include <PassModule.hpp>
//#include <fomac.hpp>
//#include <qdmi.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm
{

/**
 * @class QirAnnotateUnsupportedGatesPass
 * @brief This pass inserts an "unsupported" attribute to the
 * appropriate gates after querying the target platform using QDMI.
 */
class QirAnnotateUnsupportedGatesPass : public PassModule
{
  public:
    static std::string const QIS_START;

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
