/**
 * @file QirDivisionByZero.cpp
 * @brief TODO
 */

#include "../headers/QirDivisionByZero.hpp"

using namespace llvm;

/**
 * @var EC_REPORT_FUNCTION
 * @brief TODO
 */
const char *const QirDivisionByZeroPass::EC_REPORT_FUNCTION =
    "__qir__report_error_value";

/**
 * @var EC_VARIABLE_NAME
 * @brief TODO
 */
const char *const QirDivisionByZeroPass::EC_VARIABLE_NAME = "__qir__error_code";

/**
 * @var EC_QIR_DIVISION_BY_ZERO
 * @brief TODO
 */
int64_t const QirDivisionByZeroPass::EC_QIR_DIVISION_BY_ZERO = 1100;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirDivisionByZeroPass::run(Module &module,
                                             ModuleAnalysisManager &MAM)
{
    IRBuilder<> builder(module.getContext());

    module.getOrInsertGlobal(EC_VARIABLE_NAME, builder.getInt64Ty());

    error_variable_ = module.getNamedGlobal(EC_VARIABLE_NAME);

    error_variable_->setLinkage(GlobalValue::InternalLinkage);
    error_variable_->setInitializer(builder.getInt64(0));
    error_variable_->setConstant(false);

    std::vector<Instruction *> instructions;
    for (auto &function : module)
    {
        for (auto &block : function)
        {
            for (auto &instr : block)
            {
                auto *udiv = dyn_cast<SDivOperator>(&instr);
                if (udiv)
                    instructions.push_back(&instr);
            }
        }
    }

    // Injecting error code updates
    for (auto instr : instructions)
    {
        auto op2 = instr->getOperand(1);

        auto const &final_block = instr->getParent();
        auto if_block =
            final_block->splitBasicBlock(instr, "if_denominator_is_zero", true);
        auto start_block = if_block->splitBasicBlock(if_block->getTerminator(),
                                                     "-INTERMEDIATE-", true);
        start_block->takeName(final_block);
        final_block->setName("after_zero_check");

        builder.SetInsertPoint(start_block->getTerminator());

        auto cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, op2,
                                      builder.getInt64(0));
        auto old_terminator = start_block->getTerminator();

        BranchInst::Create(if_block, final_block, cmp, start_block);

        old_terminator->eraseFromParent();

        raiseError(EC_QIR_DIVISION_BY_ZERO, module, if_block->getTerminator());
    }

    // Checking error codes at end of
    Function *entry = nullptr;
    std::vector<BasicBlock *> exit_blocks;
    for (auto &function : module)
    {
        if (function.hasFnAttribute("EntryPoint"))
        {
            entry = &function;
            for (auto &block : function)
            {
                auto last = block.getTerminator();
                if (last && dyn_cast<ReturnInst>(last))
                    exit_blocks.push_back(&block);
            }
            break;
        }
    }

    if (entry)
    {
        for (auto start_block : exit_blocks)
        {
            auto if_block = start_block->splitBasicBlock(
                start_block->getTerminator(), "if_error_occurred", false);
            auto final_block = if_block->splitBasicBlock(
                if_block->getTerminator(), "exit_block", false);

            builder.SetInsertPoint(start_block->getTerminator());
            LoadInst *load =
                builder.CreateLoad(builder.getInt64Ty(), error_variable_);

            auto cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_NE, load,
                                          builder.getInt64(0));
            auto old_terminator = start_block->getTerminator();

            BranchInst::Create(if_block, final_block, cmp, start_block);
            old_terminator->eraseFromParent();

            builder.SetInsertPoint(if_block->getTerminator());

            auto fnc = module.getFunction(EC_REPORT_FUNCTION);
            std::vector<Value *> arguments;
            arguments.push_back(load);

            if (!fnc)
            {
                std::vector<Type *> types;
                types.resize(arguments.size());

                for (uint64_t i = 0; i < types.size(); ++i)
                    types[i] = arguments[i]->getType();

                auto return_type = Type::getVoidTy(module.getContext());

                FunctionType *fnc_type =
                    FunctionType::get(return_type, types, false);
                fnc = Function::Create(fnc_type, Function::ExternalLinkage,
                                       EC_REPORT_FUNCTION, module);
            }

            builder.CreateCall(fnc, arguments);
        }
    }

    return PreservedAnalyses::none();
}

void QirDivisionByZeroPass::raiseError(int64_t error_code, Module &module,
                                       Instruction *instr)
{
    IRBuilder<> builder(module.getContext());
    auto const &final_block = instr->getParent();
    auto if_block = final_block->splitBasicBlock(instr, "if_ecc_not_set", true);
    auto start_block = if_block->splitBasicBlock(if_block->getTerminator(),
                                                 "-INTERMEDIATE-", true);
    start_block->takeName(final_block);
    final_block->setName("ecc_set_finally");

    builder.SetInsertPoint(start_block->getTerminator());
    LoadInst *load = builder.CreateLoad(builder.getInt64Ty(), error_variable_);
    auto cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, load,
                                  builder.getInt64(0));

    auto old_terminator = start_block->getTerminator();
    BranchInst::Create(if_block, final_block, cmp, start_block);
    old_terminator->eraseFromParent();

    builder.SetInsertPoint(if_block->getTerminator());
    builder.CreateStore(builder.getInt64(error_code), error_variable_);
}

/**
 * @brief External function for loading the 'QirDivisionByZeroPass' as a
 * 'PassModule'.
 * @return QirDivisionByZeroPass
 */
extern "C" PassModule *loadQirPass() { return new QirDivisionByZeroPass(); }
