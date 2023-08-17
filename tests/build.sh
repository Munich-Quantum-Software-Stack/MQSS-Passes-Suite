sudo apt install -y cmake llvm
cd ../
rm -rf build/ 2> /dev/null
mkdir build/
cd build/
export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
cmake ..
cmake --build .
./apply_passes
