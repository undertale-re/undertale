#!/bin/bash
#SBATCH --job-name=pretrain-maskedlm
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --distribution=nopack
#SBATCH --output=pretrain-maskedlm-%j

DATASET=humaneval-x
TOKENIZER=$SHARE/models/tokenizer/$DATASET/latest
TRAINING=$SHARE/datasets/$DATASET-pretrain-maskedlm/training/latest
VALIDATION=$SHARE/datasets/$DATASET-pretrain-maskedlm/validation/latest
BATCH=184

srun python pretrain-maskedlm.py \
    --tokenizer $TOKENIZER \
    $TRAINING \
    --validation $VALIDATION \
    $SLURM_JOB_NAME \
    --accelerator gpu \
    --nodes $SLURM_JOB_NUM_NODES \
    --devices $SLURM_NTASKS_PER_NODE \
    --version $SLURM_JOB_ID \
    --batch-size $BATCH
