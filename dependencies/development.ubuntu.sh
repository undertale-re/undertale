#!/bin/bash

# Install production dependencies.
WORKING=$(dirname "${BASH_SOURCE[0]}")
bash $WORKING/production.ubuntu.sh
