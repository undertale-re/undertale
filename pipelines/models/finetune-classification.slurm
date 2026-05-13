#!/bin/bash
#SBATCH --job-name=finetune-classification
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --distribution=nopack
#SBATCH --output=finetune-classification-%j

PRETRAIN_DATASET=nixpkgs-legacy
FINETUNE_DATASET=cvebinarysheet
TOKENIZER=$SHARE/models/tokenizer/$PRETRAIN_DATASET/latest
TRAINING=$SHARE/datasets/$FINETUNE_DATASET-finetune-classification/training/latest
VALIDATION=$SHARE/datasets/$FINETUNE_DATASET-finetune-classification/validation/latest
PRETRAINED=$SHARE/models/maskedlm/$PRETRAIN_DATASET/latest
CLASSES=10
BATCH=64

srun python finetune-classification.py \
    --tokenizer $TOKENIZER \
    $TRAINING \
    --validation $VALIDATION \
    $SLURM_JOB_NAME \
    --classes $CLASSES \
    --pretrained $PRETRAINED \
    --accelerator gpu \
    --nodes $SLURM_JOB_NUM_NODES \
    --devices $SLURM_NTASKS_PER_NODE \
    --version $SLURM_JOB_ID \
    --batch-size $BATCH
