include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1 # Use the latest stable version or one compatible with your
                 # code
)

FetchContent_MakeAvailable(fmt)
