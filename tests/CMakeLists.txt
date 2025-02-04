cmake_minimum_required(VERSION 3.10)
project(ProjectTests)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_CXX_FLAGS "-w -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")
# Add Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Define the common libraries to link
set(COMMON_LIBRARIES
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-spin.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-common.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-nlopt.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-platform-default.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-em-default.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-ensmallen.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-mlir-runtime.so
  ${GTEST_LIBRARIES}
  pthread)

# Test executable
add_executable(test-mqss-passes testMQSSPasses.cpp)
target_link_libraries(test-mqss-passes PRIVATE ${COMMON_LIBRARIES} MQSSPasses)
target_include_directories(test-mqss-passes PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAQ_MLIR_RUNTIME_PATH})

add_executable(test-decomposition-passes testCudaQDecompositionPasses.cpp)
target_link_libraries(test-decomposition-passes PRIVATE ${COMMON_LIBRARIES})
target_include_directories(test-decomposition-passes PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAQ_MLIR_RUNTIME_PATH})

add_executable(test-transpiler testCudaQTranspiler.cpp)
target_link_libraries(test-transpiler PRIVATE ${COMMON_LIBRARIES})
target_include_directories(test-transpiler PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAQ_MLIR_RUNTIME_PATH})

add_executable(test-verification-qcec testMQSS-QCEC.cpp)
target_link_libraries(test-verification-qcec PRIVATE ${COMMON_LIBRARIES} MQT::QCEC MQSSPasses)
target_include_directories(test-verification-qcec PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${MQT_QCEC_INCLUDE_DIR}
    ${CUDAQ_MLIR_RUNTIME_PATH})

set(TEST_OUTPUT_DIR $<TARGET_FILE_DIR:test-mqss-passes>/golden-cases)
add_custom_target(copy-golden-cases ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/tests/golden-cases ${TEST_OUTPUT_DIR})

set(TEST_OUTPUT_DIR $<TARGET_FILE_DIR:test-mqss-passes>/code)
add_custom_target(copy-code-test ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/tests/code ${TEST_OUTPUT_DIR})

set(TEST_OUTPUT_DIR $<TARGET_FILE_DIR:test-mqss-passes>/qasm)
add_custom_target(copy-qasm-test ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/tests/qasm ${TEST_OUTPUT_DIR})
