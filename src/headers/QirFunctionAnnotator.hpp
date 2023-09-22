#pragma once

#include "PassModule.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

     class FunctionAnnotatorPass : public PassModule 
     {
          public:
				      
              PreservedAnalyses run(Module& module, ModuleAnalysisManager& mam);
       };

} //llvm
