### **Issue Title**
*(Brief and descriptive title summarizing the issue)*

#### **Issue Type**
*(Choose one: Implementation Discussion / Support Request / Bug Report / Code Implementation Elaboration)*

---

#### **Description**

*(Explain the issue in detail, providing context and background information.)*

---

#### **Version Information**

- **cmake:** [Version]
- **llvm:** [Version]
- **rabbitmq-server:** [Version]
- **g++:** [Version]
- **curl:** [Version]
- **libgtest-dev:** [Version]
- **nlohmann-json3-dev:** [Version]
- **clang-format:** [Version]
- **pre-commit:** [Version]
- **cmakelang:** [Version]
- **python3-pip:** [Version]
- **flex:** [Version]
- **bison:** [Version]

---

#### **Bug Description**

*(Detailed explanation of the bug)*

---

#### **Minimal Working Example (MWE)**

```cpp
// Provide a minimal code snippet that reproduces the bug
// Ensure it's concise yet complete enough to reproduce the issue

// Example:
#include "../headers/QirMyPass.hpp"

using namespace llvm;

PreservedAnalyses QirMyPass::run(Module &module, ModuleAnalysisManager &MAM)
{
    // Your pass goes here
	return PreservedAnalyses::none();
}

extern "C" PassModule *loadQirPass()
{
    return new QirMyPass();
}
```

---

#### **MWE Output**

*(Include the output or error messages obtained from running the MWE)*

---

#### **Additional Information**

*(Any additional context, logs, screenshots, or proposed solutions)*
