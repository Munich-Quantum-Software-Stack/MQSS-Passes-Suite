#pragma once

#include "llvm.hpp"
#include <PassModule.hpp>
#include <qdmi.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Python.h>

namespace llvm
{


class QirPythonAPITest : public PassModule
{
    public:
        PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM,
                          QDMI_Device dev);

};

} // namespace llvm
