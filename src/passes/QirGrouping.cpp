#include "../headers/QirGrouping.hpp"
//#include "../QirGroupingAnalysis.hpp"

using namespace llvm;

std::string const QirGroupingPass::QIS_START        = "__quantum"
                                                      "__qis_";
std::string const QirGroupingPass::READ_INSTR_START = "__quantum"
                                                      "__qis__read_";

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

void QirGroupingPass::prepareSourceSeparation(Module &module, BasicBlock *block){
    // Creating replacement blocks
    LLVMContext& context = module.getContext(); // TODO: Do we need the module here?

    post_classical_block_ = BasicBlock::Create(
		context,
		"post-quantum",
		block->getParent(),
		block);

    quantum_block_ = BasicBlock::Create(
		context, "quantum", 
		block->getParent(), 
		post_classical_block_);

    pre_classical_block_ = BasicBlock::Create(
		context, 
		"pre-quantum", 
		block->getParent(), 
		quantum_block_);

    // Storing the blocks for later processing
    quantum_blocks_.push_back(quantum_block_);
    classical_blocks_.push_back(pre_classical_block_);
    classical_blocks_.push_back(post_classical_block_);

    // Renaming the block
    pre_classical_block_->takeName(block);

    // Preparing builders
    post_classical_builder_->SetInsertPoint(post_classical_block_);
    quantum_builder_->SetInsertPoint(quantum_block_);
    pre_classical_builder_->SetInsertPoint(pre_classical_block_);

    // Replacing entry
    block->setName("exit_quantum_grouping");
    block->replaceUsesWithIf(
    	pre_classical_block_,
        [](Use& use){
            auto* phi_node = dyn_cast<PHINode>(use.getUser());
            return (phi_node == nullptr);
        });
}

void QirGroupingPass::nextQuantumCycle(Module &module, BasicBlock* block){
    auto& context = module.getContext(); // TODO: DO WE NEED module HERE?
    
    pre_classical_builder_->CreateBr(quantum_block_);
    quantum_builder_->CreateBr(post_classical_block_);

    pre_classical_block_ = post_classical_block_;

    // Creating replacement blocks
    post_classical_block_ = BasicBlock::Create(
	context, 
	"post-quantum", 
	block->getParent(), 
	block);

    quantum_block_ = BasicBlock::Create(
	context, 
	"quantum", 
	block->getParent(), 
	post_classical_block_);

    // Storing the blocks for later processing
    quantum_blocks_.push_back(quantum_block_);
    classical_blocks_.push_back(post_classical_block_);

    // Preparing builders
    post_classical_builder_->SetInsertPoint(post_classical_block_);
    quantum_builder_->SetInsertPoint(quantum_block_);
    pre_classical_builder_->SetInsertPoint(pre_classical_block_);
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
            auto user = dyn_cast<User>(val);
            ret.id    = 0;

            if(user && user->getNumOperands() == 1)
            {
                auto cst = dyn_cast<ConstantInt>(user->getOperand(0));

                if(cst)
                    ret.id = cst->getValue().getZExtValue();
            }
        }
    }

    return ret;
}

void QirGroupingPass::expandBasedOnSource(Module &module, BasicBlock *block){
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

            quantum_builder_->Insert(new_instruction);

            instruction.replaceAllUsesWith(new_instruction);
            to_delete.push_back(&instruction);
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
                post_classical_builder_->Insert(new_instr);
                instruction.replaceAllUsesWith(new_instr);
                to_delete.push_back(&instruction);

                post_quantum_instructions.insert(new_instr);
                continue;
            }

            // Post quantum section
            // Moving remaining to pre-section
            auto new_instr = instruction.clone();

            new_instr->takeName(&instruction);
            pre_classical_builder_->Insert(new_instr);

            instruction.replaceAllUsesWith(new_instr);
            to_delete.push_back(&instruction);
        }
        else
            assert(((void)"Unsupported occurring while grouping instructions", false));
    }

    pre_classical_builder_->CreateBr(quantum_block_);
    quantum_builder_->CreateBr(post_classical_block_);
    post_classical_builder_->CreateBr(block);

    deleteInstructions();
}

void QirGroupingPass::expandBasedOnDest(
    Module            &module,
    BasicBlock        *block,
    bool              move_quatum,
    std::string const &name)
{
    auto& context = module.getContext(); // TODO: DO WE NEED module HERE?
    to_delete.clear();

    auto extra_block = BasicBlock::Create(
	context, 
	"unnamed", 
	block->getParent(), 
	block);

    extra_block->takeName(block);
    block->replaceUsesWithIf(
        extra_block,
        [](Use& use){
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
        }
    }

    first_builder.CreateBr(block);

    deleteInstructions();
}

void QirGroupingPass::runBlockAnalysis(Module &module) {
    for (auto& function : module) {
        for (auto& block : function) {
            bool pure_quantum     = true;
            bool pure_measurement = true;

            // Classifying the blocks
            for (auto& instr : block) {
                auto call_instr = dyn_cast<CallBase>(&instr);
                if (call_instr != nullptr) {
                    auto f = call_instr->getCalledFunction();
                    if (f == nullptr)
                        continue;

                    auto name = static_cast<std::string>(f->getName());
                    bool is_quantum = (name.size() >= QIS_START.size() &&
                                       name.substr(0, QIS_START.size()) == QIS_START);

                    bool is_measurement = (name.size() >= READ_INSTR_START.size() &&
                                           name.substr(0, READ_INSTR_START.size()) == READ_INSTR_START);

                    if (is_measurement)
                        contains_quantum_measurement_.insert(&block);

                    if (is_quantum)
                        contains_quantum_circuit_.insert(&block);

                    pure_measurement = pure_measurement && is_measurement;
                    pure_quantum     = pure_quantum && is_quantum && !is_measurement;
                }
                else {
                    // Any other instruction is makes the block non-pure
                    pure_quantum     = false;
                    pure_measurement = false;
                }
            }

            if (pure_quantum)
                pure_quantum_instructions_.insert(&block);

            if (pure_measurement)
                pure_quantum_measurement_.insert(&block);
        }
    }
}

QirGroupingPass::Result QirGroupingPass::runGroupingAnalysis(Module &module) {
    GroupAnalysis ret;

    // Preparing analysis
    contains_quantum_circuit_.clear();
    contains_quantum_measurement_.clear();
    pure_quantum_instructions_.clear();
    pure_quantum_measurement_.clear();

    // Classifying each of the blocks
    runBlockAnalysis(module);

    for (auto& function : module) {
        for (auto& block : function) {
            bool is_pure_quantum     = pure_quantum_instructions_.find(&block) != pure_quantum_instructions_.end();
            bool is_pure_measurement = pure_quantum_measurement_.find(&block) != pure_quantum_measurement_.end();

            // Pure blocks are ignored
            if (is_pure_quantum || is_pure_measurement)
                continue;

            bool has_quantum = contains_quantum_circuit_.find(&block) != contains_quantum_circuit_.end();

            // Pure classical blocks are also ignored
            if (!has_quantum)
                continue;

            bool has_measurement = contains_quantum_measurement_.find(&block) != contains_quantum_measurement_.end();

            // Differentiating between blocks that has measurements and those that has not
            if (!has_measurement)
                ret.qc_cc_blocks.push_back(&block);
            else
                ret.qc_mc_cc_blocks.push_back(&block);
        }
    }

    return ret;
}

PreservedAnalyses QirGroupingPass::run(Module &module, ModuleAnalysisManager &mam) {
    auto result   = runGroupingAnalysis(module);
    auto &context = module.getContext();
    
    pre_classical_builder_  = std::make_shared<IRBuilder<>>(context);
    quantum_builder_        = std::make_shared<IRBuilder<>>(context);
    post_classical_builder_ = std::make_shared<IRBuilder<>>(context);

    for (auto* block : result.qc_cc_blocks) {
        quantum_blocks_.clear();
        classical_blocks_.clear();

        // First split
        expandBasedOnSource(module, block);

        // Second splits
        for (auto* readout_block : quantum_blocks_)
            expandBasedOnDest(module, readout_block, true, "readout");

        // Last classical block does not contain any loads
        classical_blocks_.pop_back();
        for (auto* load_block : classical_blocks_)
            expandBasedOnDest(module, load_block, false, "load");
    }

    for (auto* block : result.qc_mc_cc_blocks) {
        quantum_blocks_.clear();
        classical_blocks_.clear();

        // First split
        expandBasedOnSource(module, block);

        // Second splits
        for (auto* readout_block : quantum_blocks_)
            expandBasedOnDest(module, readout_block, true, "readout");

        // Last classical block does not contain any loads
        classical_blocks_.pop_back();
        for (auto* load_block : classical_blocks_)
            expandBasedOnDest(module, load_block, false, "load");
    }

    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirGroupingPass();
}
