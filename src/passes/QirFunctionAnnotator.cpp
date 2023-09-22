#include "../headers/QirFunctionAnnotator.hpp"
#include "../QirModulePassManager.hpp"


using namespace llvm;

PreservedAnalyses FunctionAnnotatorPass::run(Module& module, ModuleAnalysisManager& /*mam*/)
{
      bool changed     = false;
      QirMetadata &qirMetadata = QirModulePassManager::getInstance().getMetadata();

      	auto annotations = qirMetadata.injectedAnnotations;
       
       // Removing all function call attributes
      
	if (qirMetadata.shouldRemoveCallAttributes)
        {
            for (auto& function : module)
            {
                for (auto& block : function)
                {
                    for (auto& instr : block)
                    {
                        auto call_instr = llvm::dyn_cast<llvm::CallBase>(&instr);
                        if (!call_instr)
                        {
                          continue;
																														                    }

                        call_instr->setAttributes({});
                        changed = true;
                    }
                }
           }
        }

	// Adding replaceWith as requested
	 for (auto& function : module)
        {
             auto name = static_cast<std::string>(function.getName());
             // Adding annotation if requested
	     auto it = annotations.find(name);
	     if (it == annotations.end())
	      {
	         continue;
	       }

	     function.addFnAttr("replaceWith", it->second);
             changed = true;
         }

	  if (changed)
	  {
	         return PreservedAnalyses::none();
          }

     return PreservedAnalyses::all();
}

extern "C" PassModule* createQirPass() {
	    return new FunctionAnnotatorPass();
}
