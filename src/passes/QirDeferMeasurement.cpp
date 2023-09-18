#include "../headers/QirDeferMeasurement.hpp"

using namespace llvm;

std::string const QirDeferMeasurementPass::RECORD_INSTR_END = "_record_output";
QirDeferMeasurementPass::QirDeferMeasurementPass() {
	readout_names_.insert("__quantum__qis__m__body");
    readout_names_.insert("__quantum__qis__mz__body");
 	readout_names_.insert("__quantum__qis__reset__body");
    readout_names_.insert("__quantum__qis__read_result__body");
}

PreservedAnalyses QirDeferMeasurementPass::run(Module &module, ModuleAnalysisManager &MAM) {
    for(auto &function : module){
       for (auto& block : function) {
			// Identifying record functions
			std::vector<Instruction*> records;
			for (auto& instr : block) {
				auto call = dyn_cast<CallBase>(&instr);
				if (call != nullptr) {
					auto f = call->getCalledFunction();
					if (f != nullptr) {
						auto name = static_cast<std::string>(f->getName());
						bool is_readout =
							(name.size() >= RECORD_INSTR_END.size() &&
							 name.substr(name.size() - RECORD_INSTR_END.size(), RECORD_INSTR_END.size()) ==
								 RECORD_INSTR_END);

						if (is_readout || readout_names_.find(name) != readout_names_.end()) {
							records.push_back(&instr);
						}
					}
				}
			}

			// Moving function calls
			if (!records.empty()) {
				IRBuilder<> builder(function.getContext());
				builder.SetInsertPoint(block.getTerminator());

				for (auto instr : records) {
					auto new_instr = instr->clone();
					new_instr->takeName(instr);
					builder.Insert(new_instr);
					instr->replaceAllUsesWith(new_instr);

					if (!instr->use_empty()) {
						errs() << "\tError: unexpected uses of instruction while moving records to the bottom of the block\n";
						return PreservedAnalyses::none();
					}

					instr->eraseFromParent();
				}
			}
		} 
    }

    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirDeferMeasurementPass();
}
