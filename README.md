You can find the [documentation here](https://lrz-qct-qis.gitlabpages.devweb.mwn.de/quantum_intermediate_representation/qir_passes/files.html)

![Alt](flowcharts/flow.png)

Install the following dependencies:
```bash
$ sudo apt install -y cmake llvm libopenmpi-dev g++
```

Clone this repository and move to the right branch:
```bash
$ cd ${DEV_PATH}
$ git clone https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes.git
$ cd qir_passes/
$ git checkout Plugins
```

Run the QIR Pass Runner daemon:
```bash
$ cd ${DEV_PATH}/qir_passes/
$ sh build.sh
```

Run the Selector Runner daemon:
```bash
$ cd ${DEV_PATH}/qir_passes/selector/
$ sh build.sh
```

Run a test client to submit a QIR and a pass selector:
```bash
$ cd ${DEV_PATH}/qir_passes/tests/
$ sh build.sh
```

You can also update the documentation:
```bash
$ cd ${DEV_PATH}/qir_passes/
$ sh build_documentation.sh
```

