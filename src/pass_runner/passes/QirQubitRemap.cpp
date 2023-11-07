/**
 * @file QirQubitRemap.cpp
 * @brief Implementation of the 'QirQubitRemapPass' class. <a href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirQubitRemap.cpp?ref_type=heads">Go to the source code of this file.</a>
 *
 * Adapted from:
 */

#include "../headers/QirQubitRemap.hpp"
#include "../headers/QirAllocationAnalysis.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirQubitRemapPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
    for (auto &function : module) {
        QirAllocationAnalysisPass QAAP;
        FunctionAnalysisManager FAM;
        QAAP.run(function, FAM);
        AllocationAnalysis function_details = QAAP.AnalysisResult;
        
		IRBuilder<> builder{function.getContext()};

		std::unordered_map<uint64_t, uint64_t> qubits_mapping{};
		std::unordered_map<uint64_t, uint64_t> results_mapping{};

		// Re-indexing
		for (auto const& value : function_details.resource_access) {
			auto const& index = value.index;
			switch (value.type) {
				case AllocationAnalysis::QubitResource:
					if (qubits_mapping.find(index) == qubits_mapping.end())
						qubits_mapping[index] = qubits_mapping.size();
					break;
				case AllocationAnalysis::ResultResource:
					if (results_mapping.find(index) == results_mapping.end())
						results_mapping[index] = results_mapping.size();
					break;
				case AllocationAnalysis::NotResource:
					break;
			}
		}

		// Updating values
		for (auto const &value : function_details.resource_access) {
			auto const &index = value.index;
			auto       op     = value.operand;

			auto pointer_type = dyn_cast<PointerType>(op->getType());
			if (!pointer_type)
				continue;

			uint64_t remapped_index = value.index;

			switch (value.type) {
				case AllocationAnalysis::QubitResource:
					if (qubits_mapping.find(index) != qubits_mapping.end())
						remapped_index = qubits_mapping[index];
					break;
				case AllocationAnalysis::ResultResource:
					if (results_mapping.find(index) != results_mapping.end())
						remapped_index = results_mapping[index];
					break;
				case AllocationAnalysis::NotResource:
					continue;
			}

			builder.SetInsertPoint(value.used_by);

			// Removing non-null attribute if it exists as remapping may change this
			auto call_instr = dyn_cast<CallInst>(value.used_by);
			if (call_instr) {
				auto attrs   = call_instr->getAttributes();
				auto newlist = attrs.removeParamAttribute(
					function.getContext(), 
					static_cast<uint32_t>(value.operand_id),
 					Attribute::NonNull
				);
				call_instr->setAttributes(newlist);
			}

			// Creating replacement instruction
			auto idx = APInt(64, remapped_index);

			auto       new_index = ConstantInt::get(function.getContext(), idx);
			Value*     new_instr = nullptr;

			//new_instr = nullptr;
			new_instr = new IntToPtrInst(new_index, pointer_type);

			builder.Insert(new_instr);

			value.used_by->setOperand(static_cast<uint32_t>(value.operand_id), new_instr);
		}
    }

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirQubitRemapPass' as a 'PassModule'.
 * @return QirQubitRemapPass
 */
extern "C" PassModule* loadQirPass() {
    return new QirQubitRemapPass();
}
