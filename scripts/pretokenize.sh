#!/bin/bash

# "pretokenize" which means split strings (disassembly) up into white-space separated tokens
#
# $2 is path to the base dataset with disassembly that you want to pre-tokenize
# e.g., ~/undertale_shared/datasets/assemblage-windows-disassembled-rz

if [ "$1" == "local" ]; then
    python -m undertale.datasets.scripts.pretokenize \
           $2/ \
           $2-pretokenized/ \
           -e local 
else
    python -m undertale.datasets.scripts.pretokenize \
           $2/ \
           $2-pretokenized/ \
           -e slurm \
           -p 128
fi

