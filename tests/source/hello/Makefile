BUILD ?= build
CC := gcc
CFLAGS := -Wall

TARGET := $(BUILD)/hello

.PHONY: all clean

all: $(BUILD) $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET): hello.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr $(BUILD)
