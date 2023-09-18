#pragma once

#include "PassModule.hpp"

#include <functional>
#include <unordered_map>
#include <vector>

namespace llvm {

struct AllocationAnalysis
{
    enum ResourceType
    {
        NotResource,
        QubitResource,
        ResultResource
    };

    struct ResourceAccessLocation
    {
        Value*       operand{nullptr};
        ResourceType type{ResourceType::NotResource};
        uint64_t     index{static_cast<uint64_t>(-1)};
        Instruction* used_by{nullptr};
        uint64_t     operand_id{0};
    };

    using ResourceValueToId      = std::unordered_map<Value*, ResourceAccessLocation>;
    using ResourceAccessList     = std::vector<ResourceAccessLocation>;

    uint64_t largest_qubit_index{0};
    uint64_t largest_result_index{0};
    uint64_t usage_qubit_counts{0};
    uint64_t usage_result_counts{0};

    ResourceValueToId  access_map{};
    ResourceAccessList resource_access{};
};

class QirQubitRemapPass : public PassModule {
public:
    using Result                 = AllocationAnalysis;
    using BlockSet               = std::unordered_set<BasicBlock*>;
    using ResourceType           = AllocationAnalysis::ResourceType;
    using ResourceAccessLocation = AllocationAnalysis::ResourceAccessLocation;

	//enum ResourceType
    //{
    //    None,
    //    Qubit,
    //    Result
    //};

    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
    Result runAllocationAnalysis(Function &function);

private:
	bool extractResourceId(Value* value, uint64_t& return_value, ResourceType& type) const;
};

}

