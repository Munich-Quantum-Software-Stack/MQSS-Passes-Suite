/**
 * @file scheduler_round_robin.cpp 
 * @brief Implementation of a dummy scheduler.
 */

#include <string>
#include <vector>
#include <iostream>

//#include <qdmi.hpp>

/**
 * @brief The main entry point of the program.
 *
 * The Scheduler.
 *
 * @return const char *
 */
extern "C" std::string scheduler(void) {
    // Query the available platforms
    std::vector<std::string> platforms = {"Q5", "Q20"}; //qdmi_available_platforms();   
 
    std::cout << "[Scheduler]........Returning target architecture to the Scheduler Runner"
              << std::endl;

    // Choose the target architecture
    return platforms.back();
}

