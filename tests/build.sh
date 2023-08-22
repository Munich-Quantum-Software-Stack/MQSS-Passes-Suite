sudo apt install -y cmake llvm
cd ../
rm -rf build/ 2> /dev/null
mkdir build/
cd build/
cmake -DCUSTOM_MAIN_SOURCE=tests/main.cpp -DCUSTOM_EXECUTABLE_NAME=main ..
make
./main
