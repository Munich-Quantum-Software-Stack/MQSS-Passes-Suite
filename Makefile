INSTALL_PATH ?= $(HOME)
EXEC_PATH    := $(INSTALL_PATH)/bin/src
QDMI_PATH    := $(shell pwd)/qdmi
QDMI         := $(wildcard $(QDMI_PATH)/build/libqdmi.so)
BUILD_DIR    := build
DOXYGEN      := $(shell command -v doxygen 2> /dev/null)
RABBITMQ     := $(wildcard /usr/local/lib/librabbitmq.so)

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

.PHONY: install docs clean 

all: install docs clean

built_qdmi:
	@echo "QDMI is already installed. Skipping installation."

build_qdmi:
	@echo "Installing QDMI."
	cmake -B qdmi/build -S qdmi
	cmake --build qdmi/build

built_rabbitmq:
	@echo "RabbitMQ is already installed. Skipping installation."

build_rabbitmq:
	@echo "Installing RabbitMQ."
	curl -LO https://github.com/alanxz/rabbitmq-c/archive/refs/tags/v0.13.0.tar.gz
	tar -xf v0.13.0.tar.gz
	cmake -B rabbitmq-c-0.13.0/build -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DENABLE_SSL_SUPPORT=OFF -S rabbitmq-c-0.13.0
	sudo cmake --build rabbitmq-c-0.13.0/build --target install
	sudo ldconfig

configure_rabbitmq:
	@hosts_file="/etc/hosts"; \
	hostname_entry="127.0.0.1 rabbitmq"; \
	if grep -qF "$$hostname_entry" "$$hosts_file"; then \
	    echo "RabbitMQ is already configured in this system."; \
	else \
	    echo "$$hostname_entry" | cat - "$$hosts_file" > temp && sudo mv -f temp "$$hosts_file"; \
	fi

build_qrm: $(TARGET_QDMI) $(TARGET_RABBITMQ) configure_rabbitmq

set_environment_qrm: build_qrm
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
	CMAKE_PREFIX_PATH=$$(llvm-config --libdir)/cmake/llvm cmake -B$(BUILD_DIR) -DBUILD_WITH_DOCS=OFF -DCMAKE_INSTALL_PREFIX=$(INSTALL_PATH) -DCUSTOM_QDMI_PATH=$(QDMI_PATH)
	sudo cmake --build $(BUILD_DIR) --target install
	sudo ldconfig

install: qrm

built_docs:
	@echo "Doxygen is already installed. Skipping installation."

build_docs:
	@echo "Installing Doxygen."
	git clone https://github.com/doxygen/doxygen.git
	cmake -B doxygen/build -G "Unix Makefiles" -S doxygen
	cmake --build doxygen/build
	$(MAKE) -C doxygen install

docs: set_environment_qrm $(TARGET_DOXYGEN)
	CMAKE_PREFIX_PATH=$$(llvm-config --libdir)/cmake/llvm cmake -B$(BUILD_DIR) -DBUILD_WITH_DOCS=ON -DCMAKE_INSTALL_PREFIX=$(INSTALL_PATH) -DCUSTOM_QDMI_PATH=$(QDMI_PATH)
	sudo cmake --build $(BUILD_DIR) --target install
	sudo ldconfig

clean:
	rm -rf rabbitmq-c-0.13.0 v0.13.0.tar.gz doxygen

