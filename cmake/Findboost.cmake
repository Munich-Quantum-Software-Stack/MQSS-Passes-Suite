# Include FetchContent module
include(FetchContent)

# Declare Boost details
FetchContent_Declare(
  boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG master # or a specific version tag, e.g., boost-1.81.0
)

# Populate Boost
FetchContent_MakeAvailable(boost)
