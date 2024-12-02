cmake_minimum_required(VERSION 3.10)
project(ProjectTests)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Test executable
add_executable(testMQSSPasses testMQSSPasses.cpp)

target_link_libraries(testMQSSPasses 
                      ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq.so
                      ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-spin.so
                      ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-common.so
                      ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-nlopt.so
                      ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-platform-default.so
                      ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-em-default.so
                      ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-ensmallen.so
                      ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-mlir-runtime.so
                      ${GTEST_LIBRARIES}
                      MQSSPasses
                      QuakeQMapPass
                      pthread
                      )

target_include_directories(testMQSSPasses PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAQ_MLIR_RUNTIME_PATH})

set(TEST_OUTPUT_DIR $<TARGET_FILE_DIR:testMQSSPasses>/golden-cases)
add_custom_target(copy-golden-cases ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/tests/golden-cases ${TEST_OUTPUT_DIR})
