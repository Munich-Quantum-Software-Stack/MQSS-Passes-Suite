include(FetchContent)

FetchContent_Declare(
    sys-sage
    GIT_REPOSITORY git@github.com:Durganshu/sys-sage.git
    GIT_TAG qc-integration
)

FetchContent_MakeAvailable(sys-sage)

FetchContent_GetProperties(sys-sage)

set(SYS_SAGE_INCLUDE_DIRS "${sys-sage_SOURCE_DIR}/include")