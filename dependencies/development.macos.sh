#!/bin/bash

WORKING=$(dirname "${BASH_SOURCE[0]}")

# Install production dependencies.
bash $WORKING/production.macos.sh
# Install common development dependencies.
bash $WORKING/development.common.sh
