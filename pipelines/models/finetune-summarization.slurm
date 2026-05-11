#!/bin/bash
#SBATCH --job-name=finetune-summarization
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --distribution=nopack
#SBATCH --output=finetune-summarization-%j

PRETRAIN_DATASET=nixpkgs-legacy
FINETUNE_DATASET=nixpkgs-legacy
TOKENIZER=$SHARE/models/tokenizer/$PRETRAIN_DATASET/latest
TRAINING=$SHARE/datasets/$FINETUNE_DATASET-finetune-summarization/training/latest
VALIDATION=$SHARE/datasets/$FINETUNE_DATASET-finetune-summarization/validation/latest
PRETRAINED=$SHARE/models/maskedlm/$PRETRAIN_DATASET/latest
LANGUAGE_CONFIG=$SHARE/models/gpt2/config/latest
LANGUAGE_PRETRAINED=$SHARE/models/gpt2/latest
BATCH=32

srun python finetune-summarization.py \
    --tokenizer $TOKENIZER \
    --pretrained $PRETRAINED \
    --language-config $LANGUAGE_CONFIG \
    --language-pretrained $LANGUAGE_PRETRAINED \
    $TRAINING \
    --validation $VALIDATION \
    $SLURM_JOB_NAME \
    --accelerator gpu \
    --nodes $SLURM_JOB_NUM_NODES \
    --devices $SLURM_NTASKS_PER_NODE \
    --version $SLURM_JOB_ID \
    --batch-size $BATCH
