Install CMake and LLVM:
```bash
$ sudo apt install -y cmake llvm libopenmpi-dev g++
```

Clone the passes and move to the right branch:
```bash
$ cd ${DEV_PATH}
$ git clone https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes.git
$ cd qir_passes/
$ git checkout Plugins
```

Run the QIR pass runner daemon:
```bash
$ cd ${DEV_PATH}/qir_passes/
$ sh build.sh
```

Run the pass selector runner daemon:
```bash
$ cd ${DEV_PATH}/qir_passes/selector/
$ sh build.sh
```

Run the client to submit a QIR and a pass selector:
```bash
$ cd ${DEV_PATH}/qir_passes/selector/clients/
$ sh build.sh
```

