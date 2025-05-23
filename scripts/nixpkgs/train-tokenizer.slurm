#!/bin/bash
#SBATCH -o train-tokenizer-log-%j
#SBATCH --exclusive

conda activate undertale

python \
    -m undertale.models.item.tokenizer \
    ~/undertale_shared/datasets/nixpkgs-disassembled-rizin-pretokenized/ \
    item.tokenizer.json \
    -p 32
