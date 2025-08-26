#!/bin/bash
#SBATCH --job-name=basic-eval
#SBATCH --partition=gaia
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=0
#SBATCH --distribution=nopack
#SBATCH -o eval-test-log-%j

eval "$(conda shell.bash hook)"
source activate undertale

DATASET=nixpkgs-disassembled-rizin-pretraining-small
TOKENIZER=item.tokenizer.nixpkgs.json
WORKSPACE=/state/partition1/user/$USER
MODEL=nixpkgs-full-pytorch-large/checkpoints/epoch=38-train_loss=0.16-valid_f1=0.96.ckpt

srun mkdir -p $WORKSPACE
srun rsync --progress -r ~/undertale_shared/datasets/$DATASET/ $WORKSPACE/$DATASET/
srun rsync --progress ~/undertale_shared/models/item/$TOKENIZER $WORKSPACE/
srun rsync --progress ~/undertale_shared/models/item/$MODEL $WORKSPACE/

export HF_HOME=$WORKSPACE/cache

python -m undertale.evaluation.basic_eval \
    --model $WORKSPACE/$(basename $MODEL) \
    --tokenizer $WORKSPACE/$TOKENIZER \
    --dataset $WORKSPACE/$DATASET/
