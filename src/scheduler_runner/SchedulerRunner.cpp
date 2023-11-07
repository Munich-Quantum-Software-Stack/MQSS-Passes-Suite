/**
 * @file SchedulerRunner.cpp
 * @brief TODO
 */

#include "SchedulerRunner.hpp"

using namespace llvm;

// Define a mutex for protecting shared resources
//std::mutex sharedMutex;

/**
 * @brief TODO
 * @param pathScheduler TODO
 * @return const char*
 */
std::string invokeScheduler(const std::string &pathScheduler) {
    // Load the scheduler as a shared library
    std::string path = pathScheduler;

    void* lib_handle = dlopen(path.c_str(), RTLD_LAZY);
 
    if (!lib_handle) {
        std::cerr << "[qschedulerrunner_d] Error loading scheduler as a shared library: " 
                  << dlerror() 
                  << std::endl;

        return NULL;
    }

    // Dynamic loading and linking of the shared library
    typedef std::string (*SchedulerFunction)();
    SchedulerFunction scheduler = reinterpret_cast<SchedulerFunction>(dlsym(lib_handle, "scheduler"));

    if (!scheduler) {
        std::cerr << "[qschedulerrunner_d] Error finding function in shared library: " 
                  << dlerror() 
                  << std::endl;

        dlclose(lib_handle);
        return NULL;
    }

    // Call the scheduler function
    return scheduler();
}

