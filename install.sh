export LDFLAGS="-L/opt/homebrew/opt/llvm@16/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm@16/include"
export LDFLAGS="-L/opt/homebrew/opt/rabbitmq-c/lib $LDFLAGS"
export CPPFLAGS="-I/opt/homebrew/opt/rabbitmq-c/include $CPPFLAGS"
#
make INSTALL_PATH=$HOME install
