# Collection of Passes of the MQSS {#mainpage}

<!-- IMPORTANT: Keep the line above as the first line and do not remove the label above. -->
<!----------------------------------------------------------------------------
Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------- -->

<!-- The label is needed to set this page as the main page in Doxygen. -->
<!-- This file is a static page and included in the CMakeLists.txt file. -->

## MLIR Passes of the Munich Quantum Software Stack (MQSS)

<!-- Include the content of README.md between the pair of markers DOXYGEN MAIN. -->

\snippet{doc} README.md DOXYGEN MAIN

<div align="center">
  <img class="mlir-passes" alt="MLIR passes" src="mlir-passes.png" width=80%>
</div>
### How to Use this Documentation?

This documentation provides helpful information to get you started with the collection of MLIR
passes of the Munich Quantum Software Stack (MQSS).

#### General Information

The [FAQ](faq.md) page gives an overview over frequently asked questions.

#### Hands-On

[Declaring Custom Passes](templates.md) page provides an step-by-step guide to show you how you can
implement custom MLIR passes to be used into the MQSS. When you want to validate your custom MLIR
passes, the [Development Guide](guide.md) page is a good starting point.

#### Specific Information about the Implementation of this Collection

When you are interested in the details of the semantics of functions and the whole API, the
[Reference Documentation](files.html) page is the right place to look for.
