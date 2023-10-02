#include "qdmi.hpp"

std::vector<std::string> qdmi_available_platforms() {
    std::vector<std::string> platforms = {
        "Q5",
        "Q20",
    };

    return platforms;
}

std::vector<std::string> qdmi_supported_gate_set(std::string target_platform) {
    std::vector<std::string> gate_set = {
		"__quantum__qis__barrier__body",
		"__quantum__qis__ccx__body",
		"__quantum__qis__cx__body",
		"__quantum__qis__cnot__body",
		"__quantum__qis__cz__body",
		"__quantum__qis__h__body",
		"__quantum__qis__mz__body",
		"__quantum__qis__reset__body",
		"__quantum__qis__rx__body",
		"__quantum__qis__ry__body",
		"__quantum__qis__rz__body",
		"__quantum__qis__s__body",
		"__quantum__qis__s_adj__body",
		"__quantum__qis__swap__body",
		"__quantum__qis__t__body",
		"__quantum__qis__t_adj__body",
		"__quantum__qis__x__body",
		"__quantum__qis__y__body",
		"__quantum__qis__z__body",
		"__quantum__qis__if_result__body",
	};    

    return gate_set;
}

