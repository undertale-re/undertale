#!/bin/bash
#SBATCH --job-name=finetune-multilabel-classification
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --distribution=nopack
#SBATCH --output=finetune-multilabel-classification-%j

PRETRAIN_DATASET=nixpkgs-legacy
FINETUNE_DATASET=nixpkgs-legacy_10percent
TOKENIZER=$SHARE/models/tokenizer/$PRETRAIN_DATASET/latest
TRAINING=$SHARE/datasets/$FINETUNE_DATASET-finetune-multilabel-classification/training/latest
VALIDATION=$SHARE/datasets/$FINETUNE_DATASET-finetune-multilabel-classification/validation/latest
PRETRAINED=$SHARE/models/maskedlm/$PRETRAIN_DATASET/latest
BATCH=64

srun python finetune-multilabel-classification.py \
    --tokenizer $TOKENIZER \
    $TRAINING \
    --validation $VALIDATION \
    $SHARE/models/tensorboard/multilabel-classification/$FINETUNE_DATASET \
    --pretrained $PRETRAINED \
    --label-balance \
    --accelerator gpu \
    --nodes $SLURM_JOB_NUM_NODES \
    --devices $SLURM_NTASKS_PER_NODE \
    --version $SLURM_JOB_ID \
    --batch-size $BATCH
