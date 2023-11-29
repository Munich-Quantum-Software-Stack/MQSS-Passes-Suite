#include "IQMBackend.hpp"

std::unordered_map<std::string, int>
Q20Backend::run_job(std::unique_ptr<Module> &module, int n_shots)
{
    std::unordered_map<std::string, int> measurements;
    std::cout << "   [Backend].............Running job in Q20Backend for "
              << n_shots << " shots." << std::endl;

    std::chrono::seconds duration(3);
    std::this_thread::sleep_for(duration);

    measurements["00"] = 2500;
    measurements["01"] = 2500;
    measurements["10"] = 2500;
    measurements["11"] = 2500;

    return measurements;
}

int Q20Backend::close_backend()
{
    std::cout << "   [Backend].............Closing Q20Backend: " << std::endl;

    return 0;
}
