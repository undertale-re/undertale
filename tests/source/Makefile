PROJECTS := \
	42 \
	hello \
    canary \
    tword \
    data

BUILD := build

.PHONY: all clean $(PROJECTS)

all: $(PROJECTS)

$(PROJECTS):
	@echo "[+] building $@"
	$(MAKE) -C $@ BUILD=$(abspath $(BUILD))

clean:
	@for dir in $(PROJECTS); do \
		echo "[-] cleaning $$dir"; \
		$(MAKE) -C $$dir BUILD=$(abspath $(BUILD)) clean; \
	done
	@echo "[-] done"
	rm -rf $(BUILD)
