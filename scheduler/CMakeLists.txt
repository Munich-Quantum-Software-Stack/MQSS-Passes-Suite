cmake_minimum_required(VERSION 3.22)
project(QirSchedulerRunner
  VERSION 0.1
  DESCRIPTION "QIR Scheduler Runner")

# Create the qselectorrunner_d.log file
file(WRITE "$ENV{HOME}/qschedulerrunner_d.log" "")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Set the installation directory for the daemon
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation directory" FORCE)
endif()
set(SCHEDULER_RUNNER_DAEMON_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

# Set the source files for your project
set(SOURCES
    qschedulerrunner_d.cpp
    connection_handling.cpp
)

# Find the Threads package
find_package(Threads REQUIRED)

# Specify the path to QDMI using the CUSTOM_QDMI_PATH variable
set(QDMI_PATH ${CUSTOM_QDMI_PATH})

# Specify the paths to the directories containing libqdmi.so and librabbitmq.so
link_directories(${CUSTOM_QDMI_PATH}/build /usr/local/lib)

# Include the directory containing qdmi.hpp
include_directories(${CUSTOM_QDMI_PATH})

## Specify the path to the RabbitMQ library
#link_directories(/usr/local/lib)

# Build the executable
add_executable(qschedulerrunner_d ${SOURCES})

# Link the RabbitMQ and the Threads libraries
target_link_libraries(qschedulerrunner_d PRIVATE Threads::Threads rabbitmq qdmi)

# Include selectors subdirectory
add_subdirectory(schedulers)

# Include RabbitMQ header files
target_include_directories(qschedulerrunner_d PRIVATE /usr/local/include)

# Install the daemon
install(TARGETS qschedulerrunner_d DESTINATION bin)

