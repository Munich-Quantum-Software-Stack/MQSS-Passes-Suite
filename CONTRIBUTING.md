# Contributing to qir_passes

Thank you for your interest in contributing to qir_passes! We appreciate your support and welcome your contributions.

Before you get started, please take a moment to read this document to understand the process for contributing to our project.

## How to Contribute

1. **Fork the Repository**: Click the "Fork" button on the top right of the repository's page to create your own copy.

2. **Clone Your Fork**: Clone your forked repository to your local machine:
   ```shell
   git clone https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/your-fork.git
   ```

3. **Create a new branch**: Create a new branch for your contribution:
   ```shell
   git checkout -b custom-pass/name-of-pass
   ```

4. **Create a custom pass**: Once your contribution is ready you need to register the custom pass. Let `MyOptimization` be the name of your new pass. It shall then be located at `src/pass_runner/passes/QirMyOptimization.cpp`, whereas its header shall be located at `src/pass_runner/headers/QirMyOptimization.hpp`.

   - Add the name of the pass to the `src/pass_runner/passes/CMakeLists.txt` file:
      ```cmake
      set(PASSES_SOURCE_FILES
          # ...
          QirMyOptimization.cpp
          # ...
      )
      ```

   - Add the name of the pass compiled as a shared library to the selector `src/selector_runner/selectors/selector_all.cpp` for CI testing purposes:
      ```cpp
      std::vector<std::string> passes {
          // ...
          libQirMyOptimizationPass.so
          // ...
      };
      ```

5. **Test**: Ensure that your changes work as intended and don't introduce any new issues.

6. **Format**: Apply the right formatting to your code base by manually triggering `clang-format` `pre-commit` hook.
   ```shell
   make format
   ```

7. **Commit Your Changes**: Commit your changes with a clear and concise message:
   ```shell
   git commit -m "Added name-of-pass"
   ```

8. **Push to Your Fork**: Push your changes to your fork on GitHub:
   ```shell
   git push origin custom-pass/name-of-pass
   ```

9. **Create a Pull Request**: Open a pull request from your branch to the `Plugins` branch in the original repository.

10. **Discuss and Revise**: Engage in any discussions or changes requested by the maintainers.

11. **Get Your Pull Request Merged**: Once your contribution is approved, it will be merged into the project.

## Code Style and Guidelines

Before making contributions, please familiarize yourself with our coding standards and guidelines. Kindly ensure that your code contributions include comments that adhere to the formatting and conventions expected by [Doxygen](https://www.doxygen.nl/manual/docblocks.html), as clear and consistent documentation is essential for maintaining and understanding the project.

## Reporting Issues

If you encounter any bugs or issues with the project, please report them on the [Issues](https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/issues) page.

## Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) to understand our community's expectations and behavior standards.

## Questions and Assistance

If you have any questions or need assistance with your contributions, please feel free to reach out to us via [email](mailto:jorge.echavarria@lrz.de) or create an issue in the repository.

Thank you for your contributions to qir_passes!

## License

By contributing to this project, you agree to the terms and conditions of the [License](LICENSE) for qir_passes.
