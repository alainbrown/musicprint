
BUILD_DIR = build

.PHONY: all clean test

all: $(BUILD_DIR)/Makefile
	$(MAKE) -C $(BUILD_DIR)

$(BUILD_DIR)/Makefile:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ..

test: all
	cd $(BUILD_DIR) && ./test_art
	cd $(BUILD_DIR) && ./test_meta
	cd $(BUILD_DIR) && ./test_search

clean:
	rm -rf $(BUILD_DIR)
