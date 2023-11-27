#include "IQMBackend.hpp"
#include <iostream>

int IQMBackend::close_backend()
{
    std::cerr << "   [Backend]...........IQMBackend close_backend()"
              << " - should not be called directly!" << std::endl;

    return -1;
}

std::vector<int> IQMBackend::run_job(std::unique_ptr<Module> &module,
                                     int n_shots)
{
    std::cerr << "   [Backend]...........IQMBackend run_job()"
              << " - should not be called directly!" << std::endl;

    return {-1};
}

std::shared_ptr<IQMBackend> IQMBackend::create_instance(const std::string &url)
{
    std::shared_ptr<IQMBackend> instance;

    // Checking URL and creating a specific derived class
    if (url == "http://localhost:9100")
    {
        instance = std::make_shared<Q5Backend>();
    }
    else if (url == "http://localhost:9101")
    {
        instance = std::make_shared<Q20Backend>();
    }

    return instance;
}
