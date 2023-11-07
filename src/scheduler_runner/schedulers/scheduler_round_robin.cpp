/**
 * @file scheduler_round_robin.cpp 
 * @brief Implementation of a dummy scheduler.
 */

#include <string>
#include <vector>

#include <qdmi.hpp>

/**
 * @brief The main entry point of the program.
 *
 * The Scheduler.
 *
 * @return const char *
 */
std::string scheduler(void) {
    // Query the available platforms
    std::vector<std::string> platforms = qdmi_available_platforms();   
 
    // Choose the target architecture
    return platforms.back();
}

