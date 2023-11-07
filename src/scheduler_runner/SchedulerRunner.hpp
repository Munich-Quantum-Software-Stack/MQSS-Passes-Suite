/**
 * @file SchedulerRunner.hpp
 * @brief TODO
 */

#ifndef SCHEDULERRUNNER_HPP
#define SCHEDULERRUNNER_HPP

#include <iostream>
#include <csignal>
#include <sys/socket.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <mutex>
#include "llvm.hpp"

std::string invokeScheduler(const std::string &pathScheduler);

#endif // SCHEDULERRUNNER_HPP

