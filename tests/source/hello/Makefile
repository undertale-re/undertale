BUILD ?= build
CC := gcc
CFLAGS := -Wall

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
