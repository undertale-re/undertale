#!/bin/bash
#SBATCH --job-name=pretrain-h100
#SBATCH --partition=xeon-g6430-h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=0
#SBATCH --distribution=nopack
#SBATCH -o pretrain-log-%j

eval "$(conda shell.bash hook)"
conda activate undertale

DATASET=nixpkgs-disassembled-rizin-pretraining-small
TOKENIZER=item.tokenizer.nixpkgs.json
WORKSPACE=/state/partition1/user/$USER

export HF_HOME=$WORKSPACE/cache

srun mkdir -p $WORKSPACE
srun rsync --progress -r ~/undertale_shared/datasets/$DATASET/ $WORKSPACE/$DATASET/
srun rsync --progress ~/undertale_shared/models/item/$TOKENIZER $WORKSPACE/

srun python -m undertale.models.item.pretrain-maskedlm \
    --tokenizer $WORKSPACE/$TOKENIZER \
    $WORKSPACE/$DATASET/ \
    pretrain-maskedlm/ \
    --accelerator gpu \
    --nodes 2 \
    --devices 2 \
    --batch-size 144
