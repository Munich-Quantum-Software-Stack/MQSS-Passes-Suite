project(MyQirTests)

# Locate and configure the Google Test framework.
find_package(GTest REQUIRED)

set(TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

find_library(RABBITMQ_LIBRARY rabbitmq)

add_executable(QirTest ${TEST_SOURCE} ../src/connection_handling.cpp)

target_link_libraries(QirTest ${RABBITMQ_LIBRARY} ${GTEST_LIBRARIES})

# Automatically discover and run tests using CTest.
include(GoogleTest)
add_test(NAME QirTest COMMAND QirTest)
