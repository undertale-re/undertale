#!/bin/bash
#SBATCH --job-name=pretrain-h100
#SBATCH -o pretrain-log-%j-%t
#SBATCH --partition=xeon-g6430-h100
#SBATCH --exclusive
#SBATCH --gres=gpu:h100:2


eval "$(conda shell.bash hook)"
conda activate undertale

DATASET=nixpkgs-disassembled-rizin-pretraining-small
TOKENIZER=item.tokenizer.nixpkgs.json
WORKSPACE=/state/partition1/user/$USER

export HF_HOME=$WORKSPACE/cache

mkdir -p $WORKSPACE
rsync --progress -r ~/undertale_shared/datasets/$DATASET/ $WORKSPACE/$DATASET/
rsync --progress ~/undertale_shared/models/item/$TOKENIZER $WORKSPACE/

srun python -m undertale.models.item.pretrain-maskedlm \
    -t $WORKSPACE/$TOKENIZER \
    $WORKSPACE/$DATASET/ \
    pretrain-maskedlm/ \
    -b 128
