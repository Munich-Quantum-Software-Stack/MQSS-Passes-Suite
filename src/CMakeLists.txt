#file(GLOB_RECURSE mqt-qmap-qdmi_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ArchitectureFactory.cpp")

#file(GLOB_RECURSE mqt-qmap-qdmi_HEADERS "${MQT_QMAP_QDMI_INCLUDE_BUILD_DIR}/ArchitectureFactory.hpp")

#add_library(mqt-qmap-qdmi SHARED ${mqt-qmap-qdmi_SOURCES} ${mqt-qmap-qdmi_HEADERS})
#target_include_directories(mqt-qmap-qdmi PUBLIC ${MQT_QMAP_QDMI_INCLUDE_BUILD_DIR})
#target_link_libraries(
#  mqt-qmap-qdmi
#  PUBLIC mqt-qmap-heuristic qdmi
#  PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

# Allow for undefined symbols in the shared library on macOS
#if(APPLE)
#  set_target_properties(mqt-qmap-qdmi PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
#endif()

# enable PIC for shared library
#set_property(TARGET mqt-qmap-heuristic PROPERTY POSITION_INDEPENDENT_CODE ON)
#set_property(TARGET qdmi PROPERTY POSITION_INDEPENDENT_CODE ON)

# install mqt-qmap-qdmi
#install(TARGETS mqt-qmap-qdmi
#        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/passes")
