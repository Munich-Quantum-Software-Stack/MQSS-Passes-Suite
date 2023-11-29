/**
 * @file SchedulerRunner.cpp
 * @brief TODO
 */

#include "SchedulerRunner.hpp"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/**
 * @brief TODO
 * @param pathScheduler TODO
 * @return std::string
 */
int invokeScheduler(const std::string &nameScheduler)
{
    std::string pathScheduler;
    char buffer[PATH_MAX];

    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1)
    {
        buffer[len] = '\0';
        pathScheduler = std::string(buffer);
        size_t lastSlash = pathScheduler.find_last_of("/\\");
        pathScheduler = pathScheduler.substr(0, lastSlash) +
                        "/lib/scheduler_runner/schedulers/";
    }
    pathScheduler.append(nameScheduler);

    std::cout << "   [Scheduler Runner]....Invoking scheduler: "
              << nameScheduler << std::endl;

    // Load the scheduler as a shared library
    void *lib_handle = dlopen(pathScheduler.c_str(), RTLD_LAZY);

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
