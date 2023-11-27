/**
 * @file qdmi.cpp
 * @brief Implementation of a dummy QDMI.
 */

#include "qdmi.hpp"

/**
 * @brief Function that returns a vector with the
 * list of availble quantum platforms.
 * @return std::vector<std::string>
 * @todo To be implemented
 */
std::vector<std::string> qdmi_backend_available_platforms()
{
    std::vector<std::string> platforms = {
        "Q5",
        "Q20",
    };

    return platforms;
}

/**
 * @brief Function that returns a vector with the
 * supported gate set of a given target platform.
 * @param target_platform Target architecture to query.
 * @return std::vector<std::string>
 * @todo To be implemented
 */
std::vector<std::string> qdmi_supported_gate_set(std::string target_platform)
{
    std::vector<std::string> gate_set = {
        "__quantum__qis__barrier__body", "__quantum__qis__ccx__body",
        "__quantum__qis__cx__body",      "__quantum__qis__cnot__body",
        "__quantum__qis__cz__body",      "__quantum__qis__h__body",
        "__quantum__qis__mz__body",      "__quantum__qis__reset__body",
        "__quantum__qis__rx__body",      "__quantum__qis__ry__body",
        "__quantum__qis__rz__body",      "__quantum__qis__s__body",
        "__quantum__qis__s_adj__body",   "__quantum__qis__swap__body",
        "__quantum__qis__t__body",       "__quantum__qis__t_adj__body",
        "__quantum__qis__x__body",       "__quantum__qis__y__body",
        "__quantum__qis__z__body",       "__quantum__qis__if_result__body",
    };

    return gate_set;
}

/**
 * @brief TODO
 * @param TODO
 * @todo To be implemented
 */
std::shared_ptr<JobRunner> qdmi_backend_open(const std::string &target_platform)
{
    std::string url;

    if (target_platform == "Q5")
    {
        url = "http://localhost:9100";
        std::shared_ptr<IQMBackend> backend = IQMBackend::create_instance(url);

        if (!backend)
        {
            std::cerr << "Failed to open the IQM backend." << std::endl;
            return nullptr;
        }

        return std::dynamic_pointer_cast<JobRunner>(backend);
    }

    if (target_platform == "Q20")
    {
        url = "http://localhost:9101";
        std::shared_ptr<IQMBackend> backend = IQMBackend::create_instance(url);

        if (!backend)
        {
            std::cerr << "Failed to open the IQM backend." << std::endl;
            return nullptr;
        }

        return std::dynamic_pointer_cast<JobRunner>(backend);
    }

    std::cout << "Invalid target architecture." << std::endl;

    return nullptr;
}

/**
 * @brief TODO
 * @param TODO
 * @todo To be implemented
 */
std::vector<int> qdmi_launch_qir(std::shared_ptr<JobRunner> handle,
                                 std::unique_ptr<Module> &module, int n_shots)
{

    if (!module)
    {
        std::cout << "   [QDMI].............Warning: Corrupt QIR module "
                  << std::endl;
        return {-1};
    }

    if (!handle)
    {
        std::cerr << "   [QDMI].............Invalid backend handle."
                  << std::endl;
        return {-1};
    }

    return handle->run_job(module, n_shots);
}

/**
 * @brief TODO
 * @param TODO
 * @todo To be implemented
 */
int qdmi_backend_close(std::shared_ptr<JobRunner> handle)
{
    // Attempt to close the backend using the provided interface
    if (handle)
    {
        handle->close_backend();
        return 0;
    }
    else
    {
        std::cerr << "   [QDMI].............Invalid backend handle."
                  << std::endl;
        return 1;
    }
}
