Install CMake and LLVM:
```bash
$ sudo apt install -y cmake llvm libopenmpi-dev
```

Clone the passes and move to the right branch:
```bash
$ cd ${DEV_PATH}
$ git clone https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes.git
$ cd qir_passes/
$ git checkout Plugins
```

Run the example:
```bash
$ cd ${DEV_PATH}/qir_passes/
$ sh build.sh
$ cd tests/
$ sh build_client.sh
```

