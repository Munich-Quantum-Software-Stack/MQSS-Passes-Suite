sudo apt install -y cmake llvm libopenmpi-dev
rm build/qpm_d 2> /dev/null
mkdir build/ 2> /dev/null
cd build/
export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
cmake -DCUSTOM_MAIN_SOURCE=src/qpm_d.cpp -DCUSTOM_EXECUTABLE_NAME=qpm_d ..
cmake --build .
./qpm_d
