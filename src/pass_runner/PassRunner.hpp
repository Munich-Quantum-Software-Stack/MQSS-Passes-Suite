/**
 * @file PassRunner.hpp
 * @brief TODO
 */

#ifndef PASSRUNNER_HPP
#define PASSRUNNER_HPP

#include "QirPassRunner.hpp"

#include <algorithm>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <signal.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

/**
 * @var QIS_START
 * @brief Used to define the quantum prefix.
 */
const std::string QIS_START = "__quantum__qis_";

void invokePasses(std::unique_ptr<Module> &module,
                  std::vector<std::string> passes);

#endif // PASSRUNNER_HPP
