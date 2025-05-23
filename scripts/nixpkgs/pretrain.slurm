#!/bin/bash
#SBATCH --job-name=pretrain-h100
#SBATCH -o pretrain-log-%j
#SBATCH --exclusive
#SBATCH --partition=xeon-g6430-h100
#SBATCH --gres=gpu:h100:2

eval "$(conda shell.bash hook)"
conda activate undertale

python -m undertale.models.item.pretrain-maskedlm \
    -t ~/undertale_shared/models/item/item.tokenizer.nixpkgs-disassembled-rizin.json \
    ~/undertale_shared/datasets/nixpkgs-disassembled-rizin-pretraining-one/ \
    pretrain-maskedlm/ \
    -b 320
