/**
 * @file PassRunner.hpp
 * @brief TODO
 */

#ifndef PASSRUNNER_HPP
#define PASSRUNNER_HPP

#include "QirPassRunner.hpp"

#include <iostream>
#include <cstring>
#include <csignal>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>

/**
 * @var QIS_START
 * @brief Used to define the quantum prefix.
 */
const std::string QIS_START = "__quantum__qis_";

void invokePasses(std::unique_ptr<Module>  &module,
                  std::vector<std::string>  passes);

#endif // PASSRUNNER_HPP
