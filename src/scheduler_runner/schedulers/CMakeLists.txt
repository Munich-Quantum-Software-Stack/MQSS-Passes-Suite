# Include all new schedulers here
set(SCHEDULERS_SOURCE_FILES
    scheduler_round_robin.cpp
    # scheduler1.cpp
    # ...
    # schedulerN.cpp
)

# Set RPATH for the shared libraries in this directory
#set(CMAKE_INSTALL_RPATH "${INSTALL_PATH}/bin/src/scheduler_runner/schedulers")
set(CMAKE_INSTALL_RPATH "${ORIGIN}/src/scheduler_runner/schedulers")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Compile each scheduler as a shared object
foreach(src_file ${SCHEDULERS_SOURCE_FILES})
    get_filename_component(src_name ${src_file} NAME_WE)
    set(target_name "${src_name}")
    add_library(${target_name} MODULE ${src_file})
    target_link_libraries(${target_name} PRIVATE qdmi)
    #install(TARGETS ${target_name} DESTINATION "${INSTALL_PATH}/bin/src/scheduler_runner/schedulers")
    install(TARGETS ${target_name} DESTINATION "${ORIGIN}/src/scheduler_runner/schedulers")
endforeach()

