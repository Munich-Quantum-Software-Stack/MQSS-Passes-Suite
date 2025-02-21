# Writing Passes for the MQSS

<!-- IMPORTANT: Keep the line above as the first line. -->
<!----------------------------------------------------------------------------
Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

TODO: License

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------- -->

<!-- This file is a static page and included in the ./CMakeLists.txt file. -->

The following sections describe how expand this collection of MLIR passes by defining custom passes.

\tableofcontents

## Registering a new pass {#pass-definition}

To define a new pass you have to register it into the `include/Passes.hpp` file. Let say, you want to include a custom pass called `CustomExamplePass`.
The method to create the pass must be registered as follows:

```cpp
std::unique_ptr<mlir::Pass> createCustomExamplePass();
```

If the pass requires arguments, those have to be also declared into the signature of the method that creates the pass. For instance, we declare a `CustomExampleArgumentPass` that receives the argument `int value`.

```cpp
std::unique_ptr<mlir::Pass> createCustomExampleArguementPass(int value);
```

Notice that the pass has to be declared into `namespace mqss::opt` to be integrated as part of the MQSS.

## Writing a new pass {#pass-new}

This example serves as a very simple template for creating custom MLIR passes using QUAKE MLIR dialect and perform some general transformation. In this example, you can create a rewrite pattern that replaces `Hadamard` operations with `S` operations.

```cpp
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Passes.hpp"

// The pass here is simple, replace Hadamard operations with S operations.

using namespace mlir;

namespace {

struct ReplaceH : public OpRewritePattern<quake::HOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(quake::HOp hOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<quake::SOp>(
        hOp, hOp.isAdj(), hOp.getParameters(), hOp.getControls(),
        hOp.getTargets());
    return success();
  }
};

class CustomExamplePassPlugin
    : public PassWrapper<CustomExamplePassPlugin, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CustomExamplePassPlugin)

  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

  void runOnOperation() override {
    auto circuit = getOperation();
    auto ctx = circuit.getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceH>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::HOp>();
    if (failed(applyPartialConversion(circuit, target, std::move(patterns)))) {
      circuit.emitOpError("simple pass failed");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCustomExamplePass(){
  return std::make_unique<CustomExamplePassPlugin>();
}
```

In order to be integrated into this project, the file `CustomExamplePass.cpp` must be in the `src` directory of this repository.

## Building your pass {#pass-build}

After including your pass in the project, you can build it as follows:

```bash
cd build
cmake ..
make 
```

## Using your new pass {#pass-use}

Once your pass is integrated into this project. You can use it to transform any given QUAKE MLIR module. Assuming that your MLIR module is into a string named `quakeModule`.
First, you need to get the context and the module itself as follows:

```cpp
auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
mlir::MLIRContext &context = *contextPtr;
```

Apart of getting context and module, the function `extractMLIRContext` register the dialects to be used, in our case QUAKE dialect too. This tells to MLIR to recognize operations and functions belonging to QUAKE dialect.

Next, you have to declare a pass manager `mlir::PassManager` and load your custom pass `mqss::opt::createCustomExamplePass` as follows:
```cpp
// creating pass manager
mlir::PassManager pm(&context);
// Adding custom pass
pm.addNestedPass<mlir::func::FuncOp>(mqss::opt::createCustomExamplePass());
```
To apply your custom pass on the MLIR module `mlirModule`, you have to run the pass manager as follows:
```cpp
// running the pass
if(mlir::failed(pm.run(mlirModule)))
  std::runtime_error("The pass failed...");
```
If your pass is successfully applied, you can dump your module to visualize the effects of your pass in the module, as follows:
```cpp
// Convert the module to a string
std::string moduleAsString;
llvm::raw_string_ostream stringStream(moduleAsString);
// Dump module to string
mlirModule->print(stringStream);
// Printing the transformed module
std::cout << "Module after Pass\n" << moduleAsString << std::endl;
```
Finally, more examples of use of custom passes and an step-by-step guide on how to test your passes can be found in [Testing your passes](guide.md).
