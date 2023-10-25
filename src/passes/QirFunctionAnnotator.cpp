/**
 * @file QirFunctionAnnotator.cpp
 * @brief Implementation of the 'QirFunctionAnnotatorPass' class. <a href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirFunctionAnnotator.cpp?ref_type=heads">Go to the source code of this file.</a>
 *
 * Adapted from: https://github.com/qir-alliance/qat/blob/main/qir/qat/Passes/FunctionReplacementPass/FunctionAnnotatorPass.cpp
 */

#include "../headers/QirFunctionAnnotator.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
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

/**
 * @brief External function for loading the 'QirFunctionAnnotatorPass' as a 'PassModule'.
 * @return QirFunctionAnnotatorPass
 */
extern "C" PassModule* loadQirPass() {
    return new QirFunctionAnnotatorPass();
}

