BUILD ?= build
CC := gcc
CFLAGS := -Wall

TARGET := $(BUILD)/tword

.PHONY: all clean

all: $(BUILD) $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET): tword.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr $(BUILD)
