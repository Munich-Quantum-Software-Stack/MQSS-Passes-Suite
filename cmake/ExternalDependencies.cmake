# ------------------------------------------------------------------------------
# Copyright 2024 Munich Quantum Software Stack Project
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://github.com/Munich-Quantum-Software-Stack/QDMI/blob/develop/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ------------------------------------------------------------------------------

include(FetchContent)
include(CMakeDependentOption)
set(FETCH_PACKAGES "")

if(BUILD_MLIR_PASSES_DOCS)
  set(DOXYGEN_VERSION
      1.12.0
      CACHE STRING "Doxygen version")
  set(DOXYGEN_REV
      "64dfee0a4b65a4dc3687dfc6b31535a844681ffa"
      CACHE STRING "Doxygen identifier (tag, branch or commit hash)")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    FetchContent_Declare(
      Doxygen
      GIT_REPOSITORY https://github.com/doxygen/doxygen.git
      GIT_TAG ${DOXYGEN_REV}
      FIND_PACKAGE_ARGS ${DOXYGEN_VERSION})
    list(APPEND FETCH_PACKAGES Doxygen)
  else()
    find_package(Doxygen ${DOXYGEN_VERSION} QUIET)
    if(NOT Doxygen_FOUND)
      FetchContent_Declare(
        Doxygen
        GIT_REPOSITORY https://github.com/doxygen/doxygen.git
        GIT_TAG ${DOXYGEN_REV})
      list(APPEND FETCH_PACKAGES Doxygen)
    endif()
  endif()

  set(DOXYGEN_AWESOME_VERSION
      1.12.0
      CACHE STRING "Doxygen Awesome version")
  set(DOXYGEN_AWESOME_REV
      "af1d9030b3ffa7b483fa9997a7272fb12af6af4c"
      CACHE STRING "Doxygen Awesome identifier (tag, branch or commit hash)")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    FetchContent_Declare(
      doxygen-awesome-css
      GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
      GIT_TAG ${DOXYGEN_AWESOME_REV}
      FIND_PACKAGE_ARGS ${DOXYGEN_AWESOME_VERSION})
    list(APPEND FETCH_PACKAGES doxygen-awesome-css)
  else()
    find_package(doxygen-awesome-css ${DOXYGEN_AWESOME_VERSION} QUIET)
    if(NOT doxygen-awesome-css_FOUND)
      FetchContent_Declare(
        doxygen-awesome-css
        GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
        GIT_TAG ${DOXYGEN_AWESOME_REV})
      list(APPEND FETCH_PACKAGES doxygen-awesome-css)
    endif()
  endif()
endif()

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
