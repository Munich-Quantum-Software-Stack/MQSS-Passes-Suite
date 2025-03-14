include(FetchContent)

FetchContent_Declare(
  qinfo
  GIT_REPOSITORY https://github.com/Munich-Quantum-Software-Stack/QInfo.git
  GIT_TAG develop)

FetchContent_MakeAvailable(qinfo)
