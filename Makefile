INSTALL_PATH ?= $(HOME)
EXEC_PATH    := $(INSTALL_PATH)/bin/lib
QDMI_PATH    := $(CURDIR)/qdmi
BUILD_DIR    := build

QDMI         := $(wildcard $(QDMI_PATH)/build/libqdmi.so)
QRM          := $(wildcard $(INSTALL_PATH)/bin/daemon_d)
RABBITMQ     := $(wildcard /usr/local/lib/librabbitmq.so)
DOXYGEN      := $(shell command -v doxygen 2> /dev/null)

ifdef QRM
	TARGET_QRM := built_qrm
else
	TARGET_QRM := build_qrm
endif

ifdef QDMI
	TARGET_QDMI := built_qdmi
else
	TARGET_QDMI := build_qdmi
endif

ifdef RABBITMQ
	TARGET_RABBITMQ := built_rabbitmq
else
	TARGET_RABBITMQ := build_rabbitmq
endif

ifdef DOXYGEN
	TARGET_DOXYGEN := built_docs
else
	TARGET_DOXYGEN := build_docs
endif

.PHONY: install clean uninstall docs test

all: install docs clean

built_qdmi:
	@echo "QDMI is already installed. Skipping installation."

build_qdmi:
	@echo "Installing QDMI."
	cmake -B $(QDMI_PATH)/build -S $(QDMI_PATH) -DCMAKE_INSTALL_PREFIX=$(INSTALL_PATH)
	cmake --build $(QDMI_PATH)/build
	cmake --install $(QDMI_PATH)/build

built_rabbitmq:
	@echo "RabbitMQ is already installed. Skipping installation."

build_rabbitmq:
	@echo "Installing RabbitMQ."
	curl -LO https://github.com/alanxz/rabbitmq-c/archive/refs/tags/v0.13.0.tar.gz
	tar -xf v0.13.0.tar.gz
	cmake -B rabbitmq-c-0.13.0/build -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DENABLE_SSL_SUPPORT=OFF -S rabbitmq-c-0.13.0
	cmake --build rabbitmq-c-0.13.0/build --target install
	ldconfig

configure_rabbitmq:
	@hosts_file="/etc/hosts"; \
	hostname_entry="127.0.0.1 rabbitmq"; \
	if grep -qF "$$hostname_entry" "$$hosts_file"; then \
		echo "RabbitMQ is already configured in this system."; \
	else \
		echo "$$hostname_entry" | cat - "$$hosts_file" > temp && mv -f temp "$$hosts_file"; \
	fi

dependencies_qrm: $(TARGET_QDMI) $(TARGET_RABBITMQ) configure_rabbitmq

qrm: dependencies_qrm
	export LD_LIBRARY_PATH="$(QDMI_PATH)/build:\
		$(EXEC_PATH)/pass_runner:\
		$(EXEC_PATH)/pass_runner/passes:\
		$(EXEC_PATH)/selector_runner:\
		$(EXEC_PATH)/selector_runner/selectors:\
		$(EXEC_PATH)/scheduler_runner:\
		$(EXEC_PATH)/scheduler_runner/schedulers:$$LD_LIBRARY_PATH" && \
	CMAKE_PREFIX_PATH=$$(llvm-config --libdir)/cmake/llvm cmake -B$(BUILD_DIR) \
		-DBUILD_WITH_DOCS=OFF \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_PATH) \
		-DCUSTOM_QDMI_PATH=$(QDMI_PATH) && \
	cmake --build $(BUILD_DIR) --target install --config Release && \
	ldconfig

install: qrm
	@echo ""
	@echo "Please add $(INSTALL_PATH)/bin to your PATH variable."
	@echo ""

clean:
	@rm -rf rabbitmq-c-0.13.0 v0.13.0.tar.gz doxygen 

uninstall: clean
	@if [ -d "$(BUILD_DIR)" ]; then \
    	cd build && make uninstall && cd ..; \
    	rm -rf $(BUILD_DIR); \
	fi; \
	rm -rf $(QDMI_PATH)/build
	@echo "Quantum Resource Manager uninstalled successfully"

built_docs:
	@echo "Doxygen is already installed. Skipping installation."

build_docs:
	@echo "Installing Doxygen."
	git clone https://github.com/doxygen/doxygen.git
	cmake -B doxygen/build -G "Unix Makefiles" -S doxygen
	cmake --build doxygen/build
	$(MAKE) -C doxygen install

docs: dependencies_qrm $(TARGET_DOXYGEN)
	export LD_LIBRARY_PATH="$(QDMI_PATH)/build:\
		$(EXEC_PATH)/pass_runner:\
		$(EXEC_PATH)/pass_runner/passes:\
		$(EXEC_PATH)/selector_runner:\
		$(EXEC_PATH)/selector_runner/selectors:\
		$(EXEC_PATH)/scheduler_runner:\
		$(EXEC_PATH)/scheduler_runner/schedulers:$$LD_LIBRARY_PATH" && \
	CMAKE_PREFIX_PATH=$$(llvm-config --libdir)/cmake/llvm cmake -B$(BUILD_DIR) \
		-DBUILD_WITH_DOCS=ON \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_PATH) \
		-DCUSTOM_QDMI_PATH=$(QDMI_PATH) && \
	cmake --build $(BUILD_DIR) --target install --config Release && \
	ldconfig

	@echo ""
	@echo "Please add $(INSTALL_PATH)/bin to your PATH variable."
	@echo ""

built_qrm: dependencies_qrm
	@echo "Quantum Resource Manager is already installed. Skipping installation."

build_qrm: qrm

run: $(TARGET_QRM)
	@if [ "$$(echo $$PATH | tr ':' '\n' | grep -c "$(INSTALL_PATH)/bin")" -eq 0 ]; then \
    	export PATH=$$PATH:$(INSTALL_PATH)/bin; \
	fi; \
	daemon_d screen

test: run
	@g++ tests/test.cpp src/connection_handling.cpp -o ./tests/test -I./src -lrabbitmq || (echo "Compilation failed"; exit 1); \
	./tests/test

