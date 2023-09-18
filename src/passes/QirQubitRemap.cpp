#include "../headers/QirQubitRemap.hpp"

using namespace llvm;

bool QirQubitRemapPass::extractResourceId(Value* value, uint64_t& return_value, ResourceType& type) const
{
    auto* instruction_ptr = dyn_cast<IntToPtrInst>(value);
    auto* operator_ptr    = dyn_cast<ConcreteOperator<Operator, Instruction::IntToPtr>>(value);

    auto* nullptr_cast = dyn_cast<ConstantPointerNull>(value);

    if (instruction_ptr || operator_ptr || nullptr_cast) {
        if (!value->getType()->isPointerTy())
            return false;

        Type* element_type = value->getType()->getPointerElementType();

        if (!element_type->isStructTy())
            return false;

        type           = ResourceType::NotResource;
        auto type_name = static_cast<std::string>(element_type->getStructName());

        if (type_name == "Qubit")
            type = ResourceType::QubitResource;

        if (type_name == "Result")
            type = ResourceType::ResultResource;

        if (type == ResourceType::NotResource)
            return false;

        bool     is_constant_int = nullptr_cast != nullptr;
        uint64_t n               = 0;

        auto user = dyn_cast<User>(value);

        // In case there exists a user, it must have exactly one argument
        // which should be an integer. In case of deferred integers, the mapping
        // will not work
        if (user && user->getNumOperands() == 1) {
            auto cst = dyn_cast<ConstantInt>(user->getOperand(0));

            if (cst) {
                is_constant_int = true;
                n               = cst->getValue().getZExtValue();
            }
        }

        if (is_constant_int) {
            return_value = n;
            return true;
        }
    }

    return false;
}

QirQubitRemapPass::Result QirQubitRemapPass::runAllocationAnalysis(Function &function) {
	AllocationAnalysis ret;

    std::unordered_set<uint64_t> qubits_used{};
    std::unordered_set<uint64_t> results_used{};

    for (auto& block : function) {
        for (auto& instr : block) {
            for (uint64_t i = 0; i < instr.getNumOperands(); ++i) {
                auto         op   = instr.getOperand(static_cast<uint32_t>(i));
                ResourceType type = ResourceType::NotResource;
                uint64_t     n    = 0;

                // Checking if it is a qubit resource reference and extracting the corresponding
                // id and type. Otherwise, we skip to the next operand.
                if (!extractResourceId(op, n, type))
                    continue;

                ResourceAccessLocation value{op, type, n, &instr, i};
                ret.access_map[op] = value;
                ret.resource_access.push_back(value);

                switch (type)
                {
					case ResourceType::QubitResource:
						qubits_used.insert(n);
						ret.largest_qubit_index = ret.largest_qubit_index < n ? n : ret.largest_qubit_index;

						break;
					case ResourceType::ResultResource:
						results_used.insert(n);
						ret.largest_result_index = ret.largest_result_index < n ? n : ret.largest_result_index;
						break;
					case ResourceType::NotResource:
						break;
                }
            }
        }
    }

    ret.usage_qubit_counts  = qubits_used.size();
    ret.usage_result_counts = results_used.size();

    return ret;	
}

PreservedAnalyses QirQubitRemapPass::run(Module &module, ModuleAnalysisManager &MAM) {
    for (auto &function : module) {
       	auto function_details = runAllocationAnalysis(function); 
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
		for (auto const& value : function_details.resource_access) {
			auto const& index = value.index;
			auto        op    = value.operand;

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

extern "C" PassModule* createQirPass() {
    return new QirQubitRemapPass();
}
