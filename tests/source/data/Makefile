BUILD ?= build
CC := gcc
CFLAGS := -Wall -O0 -fno-inline -fno-omit-frame-pointer -no-pie -g

TARGET := $(BUILD)/data

.PHONY: all clean

all: $(BUILD) $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET): data.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr $(BUILD)
