# Declare all external dependencies and make sure that they are available.

include(FetchContent)
set(FETCH_PACKAGES "")

# Find jansson package
set(JANSSON_VERSION
    2.14
    CACHE STRING "jansson version")
set(JANSSON_URL https://github.com/akheron/jansson/releases/download/v${JANSSON_VERSION}/jansson-${JANSSON_VERSION}.tar.gz)
set(JANSSON_BUILD_DOCS OFF CACHE BOOL "" FORCE)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(jansson URL ${JANSSON_URL} FIND_PACKAGE_ARGS ${JANSSON_VERSION})
  list(APPEND FETCH_PACKAGES jansson)
else()
  find_package(jansson ${JANSSON_VERSION} QUIET)
  if(NOT jansson_FOUND)
    FetchContent_Declare(jansson URL ${JANSSON_URL})
    list(APPEND FETCH_PACKAGES jansson)
  endif()
endif()

# Find FoMaC package
set(FOMAC_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../fomac")
if (EXISTS "${FOMAC_SOURCE_DIR}/CMakeLists.txt")
  set(FETCHCONTENT_SOURCE_DIR_FOMAC
      ${FOMAC_SOURCE_DIR}
      CACHE
        PATH
        "Path to the source directory of the library. This variable is used by FetchContent to download the library if it is not already available."
  )
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(fomac GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/FoMaC.git GIT_TAG develop)
  list(APPEND FETCH_PACKAGES fomac)
else()
  find_package(fomac QUIET)
  if(NOT fomac_FOUND)
    FetchContent_Declare(fomac GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/FoMaC.git GIT_TAG develop)
    list(APPEND FETCH_PACKAGES fomac)
  endif()
endif()

# Find QDMI package
set(QDMI_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../qdmi")
if (EXISTS "${QDMI_SOURCE_DIR}/CMakeLists.txt")
  set(FETCHCONTENT_SOURCE_DIR_QDMI
      ${QDMI_SOURCE_DIR}
      CACHE
        PATH
        "Path to the source directory of the library. This variable is used by FetchContent to download the library if it is not already available."
  )
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(qdmi GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/QDMI.git GIT_TAG testing)
  list(APPEND FETCH_PACKAGES qdmi)
else()
  find_package(qdmi QUIET)
  if(NOT qdmi_FOUND)
    FetchContent_Declare(qdmi GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/QDMI.git GIT_TAG testing)
    list(APPEND FETCH_PACKAGES qdmi)
  endif()
endif()

# Find QInfo package
set(QINFO_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../qinfo")
if (EXISTS "${QINFO_SOURCE_DIR}/CMakeLists.txt")
  set(FETCHCONTENT_SOURCE_DIR_QINFO
      ${QINFO_SOURCE_DIR}
      CACHE
        PATH
        "Path to the source directory of the library. This variable is used by FetchContent to download the library if it is not already available."
  )
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(qinfo GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/QInfo.git GIT_TAG testing)
  list(APPEND FETCH_PACKAGES qinfo)
else()
  find_package(qinfo QUIET)
  if(NOT qinfo_FOUND)
    FetchContent_Declare(qinfo GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/QInfo.git GIT_TAG testing)
    list(APPEND FETCH_PACKAGES qinfo)
  endif()
endif()

# Find QRM package
set(QRM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../qrm")
if (EXISTS "${QRM_SOURCE_DIR}/CMakeLists.txt")
  set(FETCHCONTENT_SOURCE_DIR_QRM
      ${QRM_SOURCE_DIR}
      CACHE
        PATH
        "Path to the source directory of the library. This variable is used by FetchContent to download the library if it is not already available."
  )
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(qrm GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/QRM.git GIT_TAG wip-cda)
  list(APPEND FETCH_PACKAGES qrm)
else()
  find_package(qrm QUIET)
  if(NOT qrm_FOUND)
    FetchContent_Declare(qrm GIT_REPOSITORY git@github.com:Munich-Quantum-Software-Stack/QRM.git GIT_TAG wip-cda)
    list(APPEND FETCH_PACKAGES qrm)
  endif()
endif()

# Find the Threads package
find_package(Threads REQUIRED)

# Find QMap package
set(MQT-QMAP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/mqt-qmap")
if (EXISTS "${MQT-QMAP_SOURCE_DIR}/CMakeLists.txt")
  set(FETCHCONTENT_SOURCE_DIR_MQT-QMAP
      ${MQT-QMAP_SOURCE_DIR}
      CACHE
        PATH
        "Path to the source directory of the library. This variable is used by FetchContent to download the library if it is not already available."
  )
endif()
set(MQT-QMAP_VERSION
    2.5.1
    CACHE STRING "MQT Qmap version")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(mqt-qmap GIT_REPOSITORY https://github.com/cda-tum/mqt-qmap.git GIT_TAG v${MQT-QMAP_VERSION}
                       FIND_PACKAGE_ARGS ${MQT-QMAP_VERSION})
  list(APPEND FETCH_PACKAGES mqt-qmap)
else()
  find_package(mqt-qmap QUIET)
  if(NOT mqt-qmap_FOUND)
    FetchContent_Declare(mqt-qmap GIT_REPOSITORY https://github.com/cda-tum/mqt-qmap.git GIT_TAG v${MQT-QMAP_VERSION}
                         FIND_PACKAGE_ARGS ${MQT-QMAP_VERSION})
    list(APPEND FETCH_PACKAGES mqt-qmap)
  endif()
endif()

# Find LLVMConfig.cmake
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
