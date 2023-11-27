#include "IQMBackend.hpp"

std::vector<int> Q20Backend::run_job(std::unique_ptr<Module> &module,
                                     int n_shots)
{
    std::cout << "   [Backend]...........Running Q20Backend job with "
              << n_shots << " shots." << std::endl;

    std::chrono::seconds duration(3);
    std::this_thread::sleep_for(duration);

    return {1, 2, 3, 4, 5};
}

int Q20Backend::close_backend()
{
    std::cout << "   [Backend]...........Closing Q20Backend: " << std::endl;

    return 0;
}
