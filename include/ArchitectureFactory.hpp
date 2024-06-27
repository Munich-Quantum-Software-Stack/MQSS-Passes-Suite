#pragma once

#include "Architecture.hpp"

#include <qdmi.h>
//#include <qdmi_internal.h>

namespace mqt {
Architecture createArchitecture(QDMI_Device dev);
}
