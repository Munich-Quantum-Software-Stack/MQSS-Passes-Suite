INSTALL_PATH ?= $(HOME)
EXEC_PATH    := $(EXEC_PATH)/bin/src
QDMI_PATH := $(shell pwd)/qdmi
BUILD_DIR    := build
DOXYGEN      := $(shell command -v doxygen 2> /dev/null)
RABBITMQ     := $(wildcard /usr/local/lib/librabbitmq.so)

ifdef RABBITMQ
	TARGET_RABBITMQ := built_dependencies_rabbitmq
else
	TARGET_RABBITMQ := build_dependencies_rabbitmq
endif

ifdef DOXYGEN
	TARGET_DOXYGEN := built_dependencies_docs
else
	TARGET_DOXYGEN := build_dependencies_docs
endif

.PHONY: all $(TARGET_RABBITMQ) build_dependencies_qrm set_environment_qrm qrm install $(TARGET_DOXYGEN) docs

all: install docs

built_dependencies_rabbitmq:
	@echo "RabbitMQ is already installed. Skipping installation."

build_dependencies_rabbitmq:
	@echo "Installing RabbitMQ."
	curl -LO https://github.com/alanxz/rabbitmq-c/archive/refs/tags/v0.13.0.tar.gz
	tar -xf v0.13.0.tar.gz
	cd rabbitmq-c-0.13.0/
	mkdir -p build
	cd build
	cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DENABLE_SSL_SUPPORT=OFF ..
	sudo cmake --build . --target install
	sudo ldconfig
	cd ../..
	rm -rf rabbitmq-c-0.13.0/

build_dependencies_qrm: $(TARGET_RABBITMQ)
	bash $(QDMI_PATH)/build.sh
	@hosts_file="/etc/hosts"; \
	hostname_entry="127.0.0.1 rabbitmq"; \
	if grep -qF "$$hostname_entry" "$$hosts_file"; then \
		echo "RabbitMQ is already configured in this system."; \
	else \
		echo "$$hostname_entry" | cat - "$$hosts_file" > temp && sudo mv -f temp "$$hosts_file"; \
	fi

set_environment_qrm: build_dependencies_qrm
	@export LD_LIBRARY_PATH=$(QDMI_PATH)/build:$$LD_LIBRARY_PATH; \
	export LD_LIBRARY_PATH=$(EXEC_PATH)/pass_runner:$$LD_LIBRARY_PATH; \
	export LD_LIBRARY_PATH=$(EXEC_PATH)/pass_runner/passes:$$LD_LIBRARY_PATH; \
	export LD_LIBRARY_PATH=$(EXEC_PATH)/selector_runner:$$LD_LIBRARY_PATH; \
	export LD_LIBRARY_PATH=$(EXEC_PATH)/selector_runner/selectors:$$LD_LIBRARY_PATH; \
	export LD_LIBRARY_PATH=$(EXEC_PATH)/scheduler_runner:$$LD_LIBRARY_PATH; \
	export LD_LIBRARY_PATH=$(EXEC_PATH)/scheduler_runner/schedulers:$$LD_LIBRARY_PATH
	
	@if [ "$$(echo $$PATH | tr ':' '\n' | grep -c "$(INSTALL_PATH)/bin")" -eq 0 ]; then \
		export PATH=$$PATH:$(INSTALL_PATH)/bin; \
	fi;

qrm: set_environment_qrm
	cmake -B$(BUILD_DIR) -DCMAKE_EXEC_PATH=$(EXEC_PATH) -DCUSTOM_QDMI_PATH=$(QDMI_PATH) -DCMAKE_PREFIX_PATH=$$(llvm-config --libdir)/cmake/llvm
	sudo cmake --build $(BUILD_DIR) --target install
	sudo ldconfig

install: qrm

built_dependencies_docs:
	@echo "Doxygen is already installed. Skipping installation."

build_dependencies_docs:
	@echo "Installing Doxygen."
	git clone https://github.com/doxygen/doxygen.git
	cd doxygen
	mkdir -p build
	cd build
	cmake -G "Unix Makefiles" ..
	make
	sudo make install
	cd ../..
	rm -rf doxygen

docs: $(TARGET_DOXYGEN)
	@echo "Building documentation with Doxygen."
	doxygen Doxyfile
	
