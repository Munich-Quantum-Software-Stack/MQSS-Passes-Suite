include(FetchContent)

FetchContent_Declare(
     mqt-qcec
     GIT_REPOSITORY https://github.com/cda-tum/mqt-qcec.git
     GIT_TAG v2.8.1
)
 
FetchContent_MakeAvailable(mqt-qcec)
 
FetchContent_GetProperties(mqt-qcec)
 
set(QCEC_INCLUDE_DIRS "${mqt-qcec_SOURCE_DIR}/include")
