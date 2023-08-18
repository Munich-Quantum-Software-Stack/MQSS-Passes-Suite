#include <llvm/Support/raw_ostream.h>

#include "../headers/QirGrouping.h"

#include <unordered_set>

using namespace llvm;

bool QirGroupingPass::isQuantumRegister(Type const *type){
    if(type->isPointerTy()){
        auto element_type = type->getPointerElementType(); // TODO: getPointerElementType IS DEPRECATED
        if(element_type->isStructTy()){
            auto type_name = static_cast<std::string const>(element_type->getStructName());
            return quantum_register_types.find(type_name) != quantum_register_types.end();
        }
    }

    return false;
}

int64_t QirGroupingPass::classifyInstruction(Instruction const *instruction){
	int64_t ret = PureClassical;

    // TODO: HOW ARE WE OBTAINING PROFILES?
    /*auto irreversible_operations = config_.irreversibleOperations();*/

    // Checking all operations
    bool any_quantum         = false;
    bool any_classical       = false;
    bool is_void             = instruction->getType()->isVoidTy();
    bool returns_quantum     = isQuantumRegister(instruction->getType());
    bool destructive_quantum = false;

    auto call_instruction = dyn_cast<CallBase>(instruction);
    if(call_instruction != nullptr){
        auto called_function = call_instruction->getCalledFunction();
        if(called_function == nullptr)
            assert(((void)"Wrong pointer", called_function != nullptr));

        // Checking if it is an irreversabile operation
        auto name = static_cast<std::string>(called_function->getName());
        if(irreversible_operations.find(name) != irreversible_operations.end())
            destructive_quantum = true;

        for(auto& arg : call_instruction->args()){
            auto q = isQuantumRegister(arg->getType());
            any_quantum |= q;
            any_classical |= !q;
        }

        if(returns_quantum || (is_void && !any_classical && any_quantum))
            ret |= DestQuantum;
    }
    else{
        for(auto& operand : instruction->operands()){
            auto q = isQuantumRegister(operand->getType());
            any_quantum |= q;
            any_classical |= !q;
        }

        // Setting the destination platform
        if (returns_quantum){
            ret |= DestQuantum;

            // If no classical or quantum arguments present, then destination dictates
            // source
            if (!any_quantum && !any_classical)
                ret |= SourceQuantum;
        }
    }

    if (destructive_quantum)
        return TransferQuantumToClassical;

    if (any_quantum && any_classical){
        if (ret != DestQuantum)
            ret = InvalidMixedLocation;
    }
    else if(any_quantum)
        ret |= SourceQuantum;

    return ret;
}

void QirGroupingPass::deleteInstructions(){
    for(auto it = to_delete.rbegin(); it != to_delete.rend(); ++it){
        auto ptr = *it;
        if(!ptr->use_empty())
	    throw std::runtime_error("Error: Unable to delete instruction.\n");
	else
            ptr->eraseFromParent();
    }
}

void QirGroupingPass::prepareSourceSeparation(Module *module, BasicBlock *block){
    // Creating replacement blocks
    LLVMContext& context = module->getContext(); // TODO: Do we need the module here?

    post_quantum_block = BasicBlock::Create(
	context,
	"post-quantum",
	block->getParent(),
	block);

    quantum_block = BasicBlock::Create(
	context, "quantum", 
	block->getParent(), 
	post_quantum_block);

    pre_quantum_block = BasicBlock::Create(
	context, 
	"pre-quantum", 
	block->getParent(), 
	quantum_block);

    // Storing the blocks for later processing
    quantum_blocks.push_back(quantum_block);
    classical_blocks.push_back(pre_quantum_block);
    classical_blocks.push_back(post_quantum_block);

    // Renaming the block
    pre_quantum_block->takeName(block);

    // Preparing builders
    post_quantum_builder->SetInsertPoint(post_quantum_block);
    quantum_builder->SetInsertPoint(quantum_block);
    pre_quantum_builder->SetInsertPoint(pre_quantum_block);

    // Replacing entry
    block->setName("exit_quantum_grouping");
    block->replaceUsesWithIf(
    	pre_quantum_block,
        [](Use& use){
            auto* phi_node = dyn_cast<PHINode>(use.getUser());
            return (phi_node == nullptr);
        });
}

void QirGroupingPass::nextQuantumCycle(Module *module, BasicBlock* block){
    auto& context = module->getContext(); // TODO: DO WE NEED module HERE?
    pre_quantum_builder->CreateBr(quantum_block);
    quantum_builder->CreateBr(post_quantum_block);

    pre_quantum_block = post_quantum_block;

    // Creating replacement blocks
    post_quantum_block = BasicBlock::Create(
	context, 
	"post-quantum", 
	block->getParent(), 
	block);

    quantum_block = BasicBlock::Create(
	context, 
	"quantum", 
	block->getParent(), 
	post_quantum_block);

    // Storing the blocks for later processing
    quantum_blocks.push_back(quantum_block);
    classical_blocks.push_back(post_quantum_block);

    // Preparing builders
    post_quantum_builder->SetInsertPoint(post_quantum_block);
    quantum_builder->SetInsertPoint(quantum_block);
    pre_quantum_builder->SetInsertPoint(pre_quantum_block);
}

QirGroupingPass::ResourceAnalysis QirGroupingPass::operandAnalysis(Value* val) const{
    // Determining if this is a static resource
    auto* instruction_ptr = dyn_cast<IntToPtrInst>(val);
    auto* operator_ptr    = dyn_cast<ConcreteOperator<Operator, Instruction::IntToPtr>>(val);
    auto* nullptr_cast    = dyn_cast<ConstantPointerNull>(val);

    ResourceAnalysis ret{};
    ret.is_const =  (instruction_ptr != nullptr) 
		 || (operator_ptr != nullptr) 
		 || (nullptr_cast != nullptr);

    // Extracting the type and index
    if(!val->getType()->isPointerTy())
        return ret;

    Type *element_type = val->getType()->getPointerElementType();

    if(!element_type->isStructTy())
        return ret;

    if(ret.is_const){
        auto type_name = static_cast<std::string>(element_type->getStructName());

        if(type_name == "Qubit")
            ret.type = ResourceType::QUBIT;
        else if(type_name == "Result")
            ret.type = ResourceType::RESULT;

        if(ret.type != ResourceType::UNDEFINED){
            auto user = llvm::dyn_cast<llvm::User>(val);
            ret.id    = 0;

            if(user && user->getNumOperands() == 1)
            {
                auto cst = llvm::dyn_cast<llvm::ConstantInt>(user->getOperand(0));

                if(cst)
                    ret.id = cst->getValue().getZExtValue();
            }
        }
    }

    return ret;
}

void QirGroupingPass::expandBasedOnSource(Module *module, BasicBlock *block){
    prepareSourceSeparation(module, block);

    // Variables used for the modifications
    to_delete.clear();
    std::unordered_set<Value*>   depends_on_qc;
    bool                         destruction_sequence_begun = false;
    std::unordered_set<Value*>   destroyed_resources{};
    std::unordered_set<uint64_t> destroyed_qubits{};
    std::unordered_set<uint64_t> destroyed_results{};

    std::unordered_set<Value*> post_quantum_instructions{};

    for(Instruction &instruction : *block){
        // Ignoring terminators
        // Only the terminator survives in the tail block
        if (instruction.isTerminator())
            continue;

        auto instr_class = classifyInstruction(&instruction);
        if ((instr_class & SourceQuantum) != 0){
            // Checking if we are starting a new quantum program
            for(auto& op : instruction.operands()){
                if(post_quantum_instructions.find(op) != post_quantum_instructions.end()){
                    nextQuantumCycle(module, block);
                    depends_on_qc.clear();
                    destroyed_resources.clear();
                    destroyed_qubits.clear();
                    destroyed_results.clear();
                    post_quantum_instructions.clear();
                    destruction_sequence_begun = false;
                    break;
                }
            }

            // Checking if the instruction is destructive
            if(instr_class == TransferQuantumToClassical){
                for(auto& op : instruction.operands()){
                    destroyed_resources.insert(op);
                    auto analysis = operandAnalysis(op);

                    // Taking note of destroyed statically allocated resources
                    if(analysis.is_const){
                        switch (analysis.type){
                            case ResourceType::QUBIT:
                                destroyed_qubits.insert(analysis.id);
                                break;
                            case ResourceType::RESULT:
                                destroyed_results.insert(analysis.id);
                                break;
                            case ResourceType::UNDEFINED:
                                break;
                        }
                    }
                }
                destruction_sequence_begun = true;
            }
            else{
                bool relies_on_destroyed_resource = false;

                for(auto& op : instruction.operands()){
                    // Skipping function pointers
                    if(dyn_cast<Function>(op))
                        continue;

                    auto analysis = operandAnalysis(op);

                    // Note that we are forced to create a new cycle if a destructive
                    // instruction is encountered. The reason is that we cannot guarantee
                    // whether a qubit reference is to a destroyed resource or not.
                    // Consider for instance, %a1.i.i = select i1 %0, %Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*)
                    // which could refer to qubit 0 or qubit 1 depending on %0. If %0 is not known
                    // at compile time, we will not be able to determine its value.

                    if(!analysis.is_const && destruction_sequence_begun){
                        relies_on_destroyed_resource = true;
                        break;
                    }

                    // If it was marked as destroyed, we break right away
                    if(destroyed_resources.find(op) != destroyed_resources.end()){
                        relies_on_destroyed_resource = true;
                        break;
                    }

                    // In case we are dealing with a constant (statically allocated)
                    // we check if the resource was destroyed.
                    if(analysis.is_const){
                        switch(analysis.type){
                            case ResourceType::QUBIT:
                                if(destroyed_qubits.find(analysis.id) != destroyed_qubits.end())
                                relies_on_destroyed_resource = true;
                                break;
                            case ResourceType::RESULT:
                                if(destroyed_results.find(analysis.id) != destroyed_results.end())
                                    relies_on_destroyed_resource = true;
                                break;
                            case ResourceType::UNDEFINED:
                                break;
                        }

                        if(relies_on_destroyed_resource)
                            break;
                    }
                }

                if(relies_on_destroyed_resource){
                    nextQuantumCycle(module, block);
                    depends_on_qc.clear();
                    post_quantum_instructions.clear();
                    destroyed_resources.clear();
                    destroyed_qubits.clear();
                    destroyed_results.clear();
                    destruction_sequence_begun = false;
                }
            }

            // Marking all instructions that depend on a a read out
            for(auto user : instruction.users())
                depends_on_qc.insert(user);

            // Moving the instruction to
            auto new_instruction = instruction.clone();
            new_instruction->takeName(&instruction);

            quantum_builder->Insert(new_instruction);

            instruction.replaceAllUsesWith(new_instruction);
            to_delete.push_back(&instruction);

	    std::string str;
            llvm::raw_string_ostream(str) << instruction;
            errs() << "Instruction to be deleted: " << str << '\n';
        }
        else if(instr_class != InvalidMixedLocation){
            // Check if depends on readout
            bool is_post_quantum_instruction = depends_on_qc.find(&instruction) != depends_on_qc.end();

            // Calls which starts with __quantum__rt__ cannot be moved to
            // the pre-calculation section becuase they might have side effects
            // such as recording output helper functions.
            auto call_instr = dyn_cast<CallBase>(&instruction);
            if(call_instr != nullptr){
                auto f = call_instr->getCalledFunction();
                if (f == nullptr)
                    continue;

                auto name = static_cast<std::string>(f->getName());
                is_post_quantum_instruction |=
                    (name.size() >= qir_runtime_prefix.size() && name.substr(0, qir_runtime_prefix.size()) == qir_runtime_prefix);
            }

            // Checking if we are inserting the instruction before or after
            // the quantum block
            if(is_post_quantum_instruction){
                for(auto user : instruction.users())
                    depends_on_qc.insert(user);

                // Inserting to post section
                auto new_instr = instruction.clone();
                new_instr->takeName(&instruction);
                post_quantum_builder->Insert(new_instr);
                instruction.replaceAllUsesWith(new_instr);
                to_delete.push_back(&instruction);

		std::string str;
                llvm::raw_string_ostream(str) << instruction;
                errs() << "Instruction to be deleted: " << str << '\n';

                post_quantum_instructions.insert(new_instr);
                continue;
            }

            // Post quantum section
            // Moving remaining to pre-section
            auto new_instr = instruction.clone();

            new_instr->takeName(&instruction);
            pre_quantum_builder->Insert(new_instr);

            instruction.replaceAllUsesWith(new_instr);
            to_delete.push_back(&instruction);

	    std::string str;
            llvm::raw_string_ostream(str) << instruction;
            errs() << "Instruction to be deleted: " << str << '\n';
        }
        else
            assert(((void)"Unsupported occurring while grouping instructions", false));
    }

    pre_quantum_builder->CreateBr(quantum_block);
    quantum_builder->CreateBr(post_quantum_block);
    post_quantum_builder->CreateBr(block);

    deleteInstructions();
}

void QirGroupingPass::expandBasedOnDest(
    Module            *module,
    BasicBlock        *block,
    bool              move_quatum,
    std::string const &name)
{
    auto& context = module->getContext(); // TODO: DO WE NEED module HERE?
    to_delete.clear();

    auto extra_block = BasicBlock::Create(
	context, 
	"unnamed", 
	block->getParent(), 
	block);

    extra_block->takeName(block);
    block->replaceUsesWithIf(
        extra_block,
        [](llvm::Use& use){
            auto* phi_node = dyn_cast<PHINode>(use.getUser());
            return (phi_node == nullptr);
        });

    block->setName(name);

    IRBuilder<> first_builder{context};
    first_builder.SetInsertPoint(extra_block);

    for(auto &instruction : *block){
        if(instruction.isTerminator())
            continue;

        auto instr_class     = classifyInstruction(&instruction);
        bool dest_is_quantum = (instr_class & DestQuantum) != 0;

        if(dest_is_quantum == move_quatum){
            auto new_instruction = instruction.clone();

            new_instruction->takeName(&instruction);
            first_builder.Insert(new_instruction);

            instruction.replaceAllUsesWith(new_instruction);
            to_delete.push_back(&instruction);

	    std::string str;
	    llvm::raw_string_ostream(str) << instruction;
	    errs() << "Instruction to be deleted: " << str << '\n';
        }
    }

    first_builder.CreateBr(block);

    deleteInstructions();
}

PreservedAnalyses QirGroupingPass::run(Module *module, ModuleAnalysisManager &mam) {
    LLVMContext &context = module->getContext();
    
    pre_quantum_builder  = std::make_shared<IRBuilder<>>(context);
    quantum_builder      = std::make_shared<IRBuilder<>>(context);
    post_quantum_builder = std::make_shared<IRBuilder<>>(context);

    std::vector<BasicBlock*> has_meas_blocks{};
    std::vector<BasicBlock*> has_no_meas_blocks{};
    std::vector<BasicBlock*> has_qc_blocks{};
    std::vector<BasicBlock*> has_init_blocks{};
    std::vector<BasicBlock*> has_rec_blocks{};

    std::vector<BasicBlock*> only_mz_blocks{};
    std::vector<BasicBlock*> only_qc_blocks{};
    std::vector<BasicBlock*> only_cc_blocks{};

    std::vector<BasicBlock*> qc_cc_blocks{};

    // Sort out blocks in different categories
    for(Function &function : *module){
        for(BasicBlock &block : function){
            bool only_cc_instructions = true;
            bool only_qc_instructions = true;
            bool only_mz_instructions = true;
            
            bool has_measurements = false;
            bool has_record = false;
            bool has_initialize = false;

            for(Instruction &instruction : block){
                CallBase *call_instruction = dyn_cast<CallBase>(&instruction);

                if(call_instruction){
                    Function *called_function = call_instruction->getCalledFunction();
                    assert(((void)"Wrong pointer", called_function != nullptr));
                    
                    std::string call_name = static_cast<std::string>(called_function->getName());
                    
                    std::string prefix("__quantum__");

                    if(!call_name.compare(0, prefix.size(), prefix)){
                        only_cc_instructions = false;
                            
                        has_measurements = has_measurements 
                                         || call_name == "__quantum__qis__mz__body";

                        only_mz_instructions =  only_mz_instructions
                                             && has_measurements/*
                                             && only_qc_instructions*/;

                        has_initialize = has_initialize
                                       || call_name == "__quantum__rt__initialize";
                            
                        has_record =  has_record
                                   || call_name == "__quantum__rt__tuple_record_output"
                                   || call_name == "__quantum__rt__result_record_output"
                                   || call_name == "__quantum__qis__read_result__body";
                    }
                    else
                        only_qc_instructions = false;
                }
            }
            
            if(has_measurements)
                has_meas_blocks.push_back(&block);
            else
                has_no_meas_blocks.push_back(&block);

            if(has_initialize)
                has_init_blocks.push_back(&block);

            if(has_record)
                has_rec_blocks.push_back(&block);

            if(only_cc_instructions)
                only_cc_blocks.push_back(&block);
            else if(only_qc_instructions){
                only_qc_blocks.push_back(&block);
                
                if(only_mz_instructions)
                    only_mz_blocks.push_back(&block);
            }
            else
                qc_cc_blocks.push_back(&block);
        }
    }

    for(BasicBlock *block : has_no_meas_blocks){
	quantum_blocks.clear();
        classical_blocks.clear();       
		
	// First split
        expandBasedOnSource(module, block);
		
	// Second splits
        for(BasicBlock *readout_block : quantum_blocks)
            expandBasedOnDest(module, readout_block, true, "readout");

	// Last classical block does not contain any loads
        classical_blocks.pop_back();
        for(BasicBlock *load_block : classical_blocks)
            expandBasedOnDest(module, load_block, false, "load");
    }

    for(BasicBlock *block : has_meas_blocks){
        quantum_blocks.clear();
        classical_blocks.clear();

        // First split
        expandBasedOnSource(module, block);

        // Second splits
        for(BasicBlock *readout_block : quantum_blocks)
            expandBasedOnDest(module, readout_block, true, "readout");

        // Last classical block does not contain any loads
        classical_blocks.pop_back();
        for(BasicBlock *load_block : classical_blocks)
            expandBasedOnDest(module, load_block, false, "load");
    }

    return PreservedAnalyses::all();
}

extern "C" PassModule* createQirGroupingPass() {
    return new QirGroupingPass();
}
