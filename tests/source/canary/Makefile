BUILD ?= build
CC := gcc
CFLAGS := -Wall -fno-omit-frame-pointer -fstack-protector-strong

TARGET := $(BUILD)/canary

.PHONY: all clean

all: $(BUILD) $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET): canary.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr $(BUILD)
