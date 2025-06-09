#!/bin/bash
#SBATCH --job-name=pretrain-h100
#SBATCH --partition=e9-h100nvl
#SBATCH --exclude=a-4-16,a-3-18,a-3-20,a-6-15,a-6-13
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
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
    --nodes $SLURM_JOB_NUM_NODES \
    --devices $SLURM_NTASKS_PER_NODE \
    --version $SLURM_JOB_ID \
    --batch-size 144
