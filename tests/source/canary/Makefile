BUILD ?= build
CC := gcc
CFLAGS := -Wall -fno-omit-frame-pointer -fstack-protector-strong

SRC := $(wildcard *.c)
NAME := $(basename $(notdir $(SRC)))
TARGET := $(BUILD)/$(NAME)

.PHONY: all clean

all: $(BUILD) $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET): $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr $(BUILD)
