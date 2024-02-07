INSTALL_PATH ?= $(HOME)
EXEC_PATH    := $(INSTALL_PATH)/bin/lib
BUILD_DIR    ?= build
DOXYGEN      := $(shell command -v doxygen 2> /dev/null)

.PHONY: install clean uninstall docs

all: install clean docs

install:
	@echo "Installing the passes."
	CMAKE_PREFIX_PATH=$$(llvm-config --libdir)/cmake/llvm cmake -B$(BUILD_DIR) \
		-DBUILD_WITH_DOCS=OFF \
        -DCMAKE_INSTALL_PREFIX=$(INSTALL_PATH) && \
	cmake --build $(BUILD_DIR) --target install --config Release && \
	if [ -n "$$CI" ]; then \
		ldconfig; \
	else \
		sudo ldconfig; \
	fi

	@echo ""
	@echo "Please add $(INSTALL_PATH) to the PASSES environment variable:"
	@echo "export PASSES=\$$PASSES:$(EXEC_PATH)/passes"
	@echo ""

clean:
	@rm -rf doxygen

uninstall: clean
	@if [ -d "$(BUILD_DIR)" ]; then \
    	cd build && make uninstall && cd ..; \
    	rm -rf $(BUILD_DIR); \
	fi;
	@echo "All passes uninstalled successfully"

ifdef DOXYGEN
build_docs:
	@echo "Doxygen is already installed. Skipping installation."
else
build_docs:
	@echo "Installing Doxygen."
	git clone https://github.com/doxygen/doxygen.git
	cmake -B doxygen/build -G "Unix Makefiles" -S doxygen
	cmake --build doxygen/build
	$(MAKE) -C doxygen install
endif

docs:
	@echo "Installing the passes."
	CMAKE_PREFIX_PATH=$$(llvm-config --libdir)/cmake/llvm cmake -B$(BUILD_DIR) \
		-DBUILD_WITH_DOCS=ON \
        -DCMAKE_INSTALL_PREFIX=$(INSTALL_PATH) && \
	cmake --build $(BUILD_DIR) --target install --config Release && \
	if [ -n "$$CI" ]; then \
		ldconfig; \
	else \
		sudo ldconfig; \
	fi

	@echo ""
	@echo "Please add $(INSTALL_PATH) to the PASSES environment variable:"
	@echo "export PASSES=\$$PASSES:$(EXEC_PATH)/passes"
	@echo ""

pre-commit:
	pre-commit run --all-files

format:
	pre-commit run clang-format --all-files
