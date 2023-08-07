Clone LLVM:
```bash
$ cd ${INST_PATH}
$ git clone --depth 1 https://github.com/llvm/llvm-project.git
$ cd llvm-project
$ git checkout release/14.x
```

Clone the passes:
```bash
$ cd ${INST_PATH}/llvm-project/llvm/lib/Transforms/Utils/
$ git submodule add https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes.git QIR
$ cd QIR
$ git checkout pass.cpp
$ git pull origin pass.cpp
```

```bash
$ cd ${INST_PATH}/llvm-project/llvm/include/llvm/Transforms/Utils/
$ git submodule add https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes.git QIR
$ cd QIR
$ git checkout header.h
$ git pull origin header.h
```

Add the following lines to llvm-project/llvm/lib/Passes/PassBuilder.cpp for every header including those you made:
```
...  
#include "llvm/Transforms/Utils/QIR/QirBarrierBeforeFinalMeasurements.h"  
#include "llvm/Transforms/Utils/QIR/QirGrouping.h"  
#include "llvm/Transforms/Utils/QIR/QirCXCancellation.h"  
#include "llvm/Transforms/Utils/QIR/new_pass.h"  
...
```

Add the following lines to llvm-project/llvm/lib/Passes/PassRegistry.def after the definition of FUNCTION_PASS:
```
...
#ifndef MODULE_PASS
#define MODULE_PASS(NAME, CREATE_PASS)
#endif
MODULE_PASS("qir-grouping", QirGroupingPass())
...
#ifndef FUNCTION_PASS
#define FUNCTION_PASS(NAME, CREATE_PASS)
#endif
FUNCTION_PASS("qir-barrier-before-final-measurements", QirBarrierBeforeFinalMeasurementsPass())
FUNCTION_PASS("qir-cx-cancellation", QirCXCancellationPass())  
FUNCTION_PASS("new-pass", new_pass())
...
```

Add the following lines to llvm-project/llvm/lib/Transforms/Utils/CMakeLists.txt:
```make
add_llvm_component_library(LLVMTransformUtils  
  QIR/QirBarrierBeforeFinalMeasurements.cpp  
  QIR/QirGrouping.cpp  
  QIR/QirCXCancellation.cpp  
  QIR/new_pass.cpp  
...
```

And build:
```bash
$ cd ${INST_PATH}/llvm-project/
$ mkdir build
$ sudo apt install cmake build-essential
$ cmake -S llvm -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
$ cd build
$ make
```
