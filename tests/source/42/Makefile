BUILD ?= build
CC := gcc
CFLAGS := -Wall

TARGET := $(BUILD)/42

.PHONY: all clean

all: $(BUILD) $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET): 42.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr $(BUILD)
