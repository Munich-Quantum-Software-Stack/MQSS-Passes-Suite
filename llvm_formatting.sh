files_list=$(find src -type f -regex '.*\.\(c\|cc\|cxx\|cpp\|h\|hpp\|hxx\)$')
clang-format --style=LLVM -i ${files_list} 
