#pragma once

#include "Architecture.hpp"
#include "qdmi.h"

namespace mqt {
Architecture createArchitecture(QDMI_Device dev);
}
