sudo apt install -y cmake llvm libopenmpi-dev
rm -rf build/ 2> /dev/null
mkdir build/
cd build/
export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
cmake -DCUSTOM_MAIN_SOURCE=src/qpm_d.cpp -DCUSTOM_EXECUTABLE_NAME=qpm_d ..
cmake --build .
./qpm_d
