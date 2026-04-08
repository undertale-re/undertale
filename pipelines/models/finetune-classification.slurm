#!/bin/bash
#SBATCH --job-name=pretrain-classification
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --distribution=nopack
#SBATCH --output=pretrain-classification-%j

DATASET=cvebinarysheet
TOKENIZER=$SHARE/models/tokenizer/nixpkgs-legacy/latest
TRAINING=$SHARE/datasets/cvebinarysheet/20260326-tokenized/tokenized-nixpkgs-training
VALIDATION=$SHARE/datasets/cvebinarysheet/20260326-tokenized/tokenized-nixpkgs-validation
BATCH=184
CHECKPOINT=$SHARE/models/maskedlm/nixpkgs-legacy/latest

srun python finetune-classification.py \
    --tokenizer $TOKENIZER \
    $TRAINING \
    ~/$SLURM_JOB_NAME \
    --accelerator gpu \
    --nodes 16 \
    --devices 2 \
    --version $SLURM_JOB_ID \
    --batch-size $BATCH \
    --checkpoint $CHECKPOINT \
    --epochs 100 \
    --generated_output_paths ~/model_outputs \
    --val_dataset $VALIDATION
