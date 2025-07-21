#!/bin/bash

# pairwise dataset which has already been pre-tokenized
#
# $1 is "local" if you are running on a Linux VM or box and "slurm" if
# you are working on the grid.
#
# $2 is path to the base dataset with disassembly that you want to tokenize
#    e.g., ~//undertale_shared/datasets/assemblage-windows-disassembled-rz
#
# This means, in fact that we have
#   input is $2-pretokenized
#   output is $2-pairs
#
# $3 is number_samples
#
# $4 is either negative_multiple
#
# Both output documents containing "input_ids" in which strings tokens
# have been replaced by numbers, and "attention_maps" which are
# mysterious.
#
#

if ! [[ "$1" =~ ^(slurm|local)$ ]];
then
    echo "Usage: pairs.sh [slurm|local] path_to_base_dataset number_samples negative_multiple"
    exit 1
fi




if [ "$1" == "local" ]; then
    python -m undertale.datasets.scripts.pairs \
           $2-pretokenized/ \
           $2-pairs/ \
           -s $3 \
           -m $4 \
           -e local 
else
    python -m undertale.datasets.scripts.pairs \
           $2-pretokenized/ \
           $2-pairs/ \
           -s $3 \
           -m $4 \
           -e slurm \
           -p 128
fi