/*
 * @file SelectorRunner.cpp
 * @brief TODO
 */

#include "SelectorRunner.hpp"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/**
 * @brief TODO
 * @param pathSelector Path to the selector to be invoked
 * @return std::vector<std::string>
 */
std::vector<std::string> invokeSelector(const std::string &nameSelector)
{
    // const char *fileName = basename(const_cast<char *>(pathSelector));

    std::string pathSelector;
    char buffer[PATH_MAX];

    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1)
    {
        buffer[len] = '\0';
        pathSelector = std::string(buffer);
        size_t lastSlash = pathSelector.find_last_of("/\\");
        pathSelector = pathSelector.substr(0, lastSlash) +
                       "/lib/selector_runner/selectors/";
    }
    pathSelector.append(nameSelector);

    std::cout << "   [Selector Runner].....Invoking selector: " << nameSelector
              << std::endl;

    // Load the selector as a shared library
    void *lib_handle = dlopen(pathSelector.c_str(), RTLD_LAZY);

    if (!lib_handle)
    {
        std::cerr << "   [Selector Runner]...Error loading selector as a "
                     "shared library: "
                  << dlerror() << std::endl;

        return std::vector<std::string>();
    }

    // Dynamic loading and linking of the shared library
    typedef std::vector<std::string> (*SelectorFunction)();
    SelectorFunction selector =
        reinterpret_cast<SelectorFunction>(dlsym(lib_handle, "selector"));

    if (!selector)
    {
        std::cerr << "   [Selector Runner]...Error finding function in shared "
                     "library: "
                  << dlerror() << std::endl;

        dlclose(lib_handle);
        return std::vector<std::string>();
    }

    // Call the selector function
    return selector();
}
