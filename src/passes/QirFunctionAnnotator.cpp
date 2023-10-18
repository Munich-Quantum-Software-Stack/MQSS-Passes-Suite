#include "../headers/QirFunctionAnnotator.hpp"

using namespace llvm;

PreservedAnalyses QirFunctionAnnotatorPass::run(Module& module, ModuleAnalysisManager& /*MAM*/) {
    QirPassRunner &QPR = QirPassRunner::getInstance();
    QirMetadata &qirMetadata = QPR.getMetadata();
      
    bool changed     = false;
    auto annotations = qirMetadata.injectedAnnotations;
       
    // Removing all function call attributes
	if (qirMetadata.shouldRemoveCallAttributes) {
        for (auto& function : module) {
            for (auto& block : function) {
                for (auto& instr : block) {
                    auto call_instr = dyn_cast<CallBase>(&instr);
                    if (!call_instr)
                          continue;
                    call_instr->setAttributes({});
                    changed = true;
                }
            }
        }
    }

	// Adding replaceWith as requested
    for (auto& function : module) {
         auto name = static_cast<std::string>(function.getName());

         // Adding annotation if requested
	     auto it = annotations.find(name);
	     if (it == annotations.end())
	         continue;

	     function.addFnAttr("replaceWith", it->second);
         changed = true;
     }

	  if (changed)
         return PreservedAnalyses::none();

     return PreservedAnalyses::all();
}

extern "C" PassModule* createQirPass() {
    return new QirFunctionAnnotatorPass();
}

