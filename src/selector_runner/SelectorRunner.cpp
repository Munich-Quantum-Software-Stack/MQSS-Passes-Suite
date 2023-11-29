/*
 * @file SelectorRunner.cpp
 * @brief TODO
 */

#include "SelectorRunner.hpp"

/**
 * @brief TODO
 * @param pathSelector Path to the selector to be invoked
 * @return std::vector<std::string>
 */
std::vector<std::string> invokeSelector(const char *pathSelector)
{
    const char *fileName = basename(const_cast<char *>(pathSelector));
    std::cout << "   [Selector Runner].....Invoking selector: " << fileName
              << std::endl;

    // Load the selector as a shared library
    void *lib_handle = dlopen(pathSelector, RTLD_LAZY);

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
