#!/bin/bash

WARNING='\033[1;33m'
END='\033[0m'

if [[ -z "${GHIDRA_INSTALL_DIR}" ]]; then
    echo -e "${WARNING}[-] cannot find Ghidra - make sure that it is installed and the GHIDRA_INSTALL_DIR environment variable is set${END}"
fi
