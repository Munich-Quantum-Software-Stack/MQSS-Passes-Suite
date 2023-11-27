/**
 * @file SchedulerRunner.cpp
 * @brief TODO
 */

#include "SchedulerRunner.hpp"

/**
 * @brief TODO
 * @param pathScheduler TODO
 * @return std::string
 */
int invokeScheduler(const std::string &pathScheduler)
{
    size_t lastSlashPos = pathScheduler.find_last_of('/');
    if (lastSlashPos != std::string::npos)
    {
        std::string fileName = pathScheduler.substr(lastSlashPos + 1);
        std::cout << "   [Scheduler Runner]..Invoking scheduler: " << fileName
                  << std::endl;
    }
    else
    {
        std::cerr << "   [Scheduler Runner]..Invalid path to scheduler"
                  << std::endl;
        return 1;
    }

    // Load the scheduler as a shared library
    std::string path = pathScheduler;

    void *lib_handle = dlopen(path.c_str(), RTLD_LAZY);

    if (!lib_handle)
    {
        std::cerr
            << "   [Scheduler Runner]..Error loading scheduler as a shared "
               "library: "
            << dlerror() << std::endl;

        return 1;
    }

    // Dynamic loading and linking of the shared library
    typedef void (*SchedulerFunction)();
    SchedulerFunction scheduler =
        reinterpret_cast<SchedulerFunction>(dlsym(lib_handle, "scheduler"));

    if (!scheduler)
    {
        std::cerr << "   [Scheduler Runner]..Error finding function in shared "
                     "library: "
                  << dlerror() << std::endl;

        dlclose(lib_handle);
    }

    // Call the scheduler function
    scheduler();

    return 0;
}
