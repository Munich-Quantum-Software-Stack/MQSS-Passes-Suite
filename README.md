# Quantum Resource Manager (QRM)

The entry point of the Quantum Resource Manager for selecting and applying LLVM passes to a Quantum Circuit described on a Quantum Intermediate Representation ([QIR](https://www.qir-alliance.org/projects/)) is the `daemon_d` daemon . This README provides instructions for compiling<!--, installing, and uninstalling the--> `daemon_d`.

## Compilation and Installation

<!--Before you can install `daemon_d`, you need to compile the project. To do this, follow the steps below:-->

To install the Quantum Resource Manager daemon system wide, follow these steps:

1. Install the required dependencies:
   - Install CMake, LLVM, RabbitMQ, and GNU's C++ frontend:
      ```bash
      sudo apt install -y cmake llvm rabbitmq-server g++
      ```

   - Download RabbitMQ-C:
      ```bash
      curl -LO https://github.com/alanxz/rabbitmq-c/archive/refs/tags/v0.13.0.tar.gz
      ```

   - Extract the file: 
      ```bash   
      tar -xf v0.13.0.tar.gz
      ```

   - Enter the extracted directory:
      ```bash
      cd rabbitmq-c-0.13.0/
      ```

   - Create a `build` directory:
      ```bash
      mkdir build/
      cd build/
      ```

   - Configure the project using CMake:
      ```bash
      cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DENABLE_SSL_SUPPORT=OFF ..
      ```

   - Build the project, return to the `qir_passes` directory, and delete the extracted directory:
      ```bash
      sudo cmake --build . --target install
      sudo ldconfig
      cd ../..
      rm -rf rabbitmq-c-0.13.0/
      ```

2. Configure the environment:
   - Set the installation path of the QRM, the location of the chose QDMI, and the path to locate LLVM's CMake configuration:
      ```bash
      export INSTALL_PREFIX=$HOME
      export QDMI_PATH=$PWD/qdmi
      export CMAKE_PREFIX_PATH=$(llvm-config --libdir)/cmake/llvm
      ```
   
   - Set the `LD_LIBRARY_PATH` environment variable with the location of the dynamic libraries:
      ```bash
      export LD_LIBRARY_PATH=$QDMI_PATH/build:$LD_LIBRARY_PATH
      export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/pass_runner:$LD_LIBRARY_PATH
      export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/pass_runner/passes:$LD_LIBRARY_PATH
      export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/selector_runner:$LD_LIBRARY_PATH
      export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/selector_runner/selectors:$LD_LIBRARY_PATH
      export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/scheduler_runner:$LD_LIBRARY_PATH
      export LD_LIBRARY_PATH=$INSTALL_PREFIX/bin/src/scheduler_runner/schedulers:$LD_LIBRARY_PATH
      ```

3. Compile the chosen Quantum Device Management Interface (QDMI), for example:
   - Navigate to the `qdmi` directory which contains a dummy QDMI:
      ```bash
      cd $QDMI_PATH
      ```

   - Create a `build` directory:
      ```bash
      mkdir build/
      cd build/
      ```

   - Configure the project using CMake:
      ```bash
      cmake ..
      ```
   - Build the project and return to the `qir_passes` directory: 
      ```bash
      make
      ```

4. Navigate to the `qir_passes` directory (if you are not already there) and create a `build` directory:
   ```bash
   mkdir build/
   cd build/
   ```

5. Configure the project using CMake, specifying the custom path for <!--installation and for --> locating the passes and the path to the chosen QDMI compiled in step 1:
   ```bash
   cmake -DCMAKE_INSTALL_PREFIX=$HOME -DCUSTOM_QDMI_PATH=qdmi ..
   ```

6. Build the project:
   ```bash
   sudo cmake --build . --target install
   sudo ldconfig
   ```

7. Configure RabbitMQ. Make sure the file `/etc/hosts` exists and contains the following line:
   ```vim
   127.0.0.1 rabbitmq
   ```

## Uninstallation

If you ever need to uninstall `daemon_d`, follow these steps:

1. Navigate to the `build` directory (if you are not already there):
   ```bash
   cd build/
   ```

2. Run the uninstall target using sudo:
   ```bash
   sudo make uninstall
   ```

This will remove `daemon_d` from your system.

## Documentation and Resources

This section provides links to project documentation and additional resources:

- [Documentation](https://lrz-qct-qis.gitlabpages.devweb.mwn.de/quantum_intermediate_representation/qir_passes/files.html): Detailed documentation about the Quantum Resource Manager.
- [Wiki](https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/wikis/home): Project wiki with additional information and guides.
- [Contributing Guidelines](CONTRIBUTING.md): Document to understand the process for contributing to our project.
<!--
- Flowchart of the QIR Pass Runner daemon: 
![Alt](flowcharts/flow.png)
-->

## Building Documentation

You can build the documentation locally using Doxygen. 

1. If Doxygen is not installed, you can install it with the following steps:

   - Install the required dependencies for Doxygen:
     ```bash
     sudo apt install -y flex bison
     ```

   - Clone the Doxygen repository:
      ```bash
      git clone https://github.com/doxygen/doxygen.git
      ```
   
   - Create a `build` directory
      ```bash
      cd doxygen
      mkdir build/
      cd build/
      ```
   
   - Run cmake with the makefile generator
      ```bash
      cmake -G "Unix Makefiles" ..
      make
      sudo make install
      cd ../..
      rm -rf doxygen
      ```

2. If Doxygen is already installed you can continue with the following steps:

   - Generate the documentation with Doxygen:
      ```bash
      doxygen Doxyfile
      ```
   
   - Open the generated documentation in a web browser:
      ```bash
      xdg-open documentation/html/index.html
      ```

Alternatively, you can manually open the file `documentation/html/index.html` with your preferred web browser.

## Running Examples

You can run the Quantum Resource Manager daemon and a test client as follows:

1. Install `daemon_d` as shown above. 
   - Then simply run the daemon specifying a path for the log file:
   ```bash
   daemon_d log ${HOME}
   ```

   - One may also run the daemon specifying the terminal as the standard output stream and no log file:
   ```bash
   daemon_d screen
   ```

2. To compile and run a test client for submitting a Quantum Circuit described in QIR to the daemon, navigate to the `tests` directory using a second terminal:
   ```bash
   cd qir_passes/tests/
   ```

3. Compile the test client:
   ```bash
   g++ test.cpp ../src/connection_handling.cpp -o test -lrabbitmq
   ```

4. Run the test client:
   ```bash
   ./test
   ```

