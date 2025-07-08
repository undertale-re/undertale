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

source /etc/profile
module purge
module load anaconda/Python-ML-2025a
# eval "$(conda shell.bash hook)"
source activate undertale
module unload anaconda/Python-ML-2025a

DATASET=nixpkgs-disassembled-rizin-pretraining-small
TOKENIZER=item.tokenizer.nixpkgs.json
WORKSPACE=/state/partition1/user/$USER
MODEL_LOC=/data2/groups/undertale_shared/models/item/nixpkgs-full-pytorch-large/checkpoints/epoch=0-train_loss=0.39-valid_f1=0.90.ckpt
export HF_HOME=$WORKSPACE/cache

srun mkdir -p $WORKSPACE
srun rsync --progress -r ~/undertale_shared/datasets/$DATASET/ $WORKSPACE/$DATASET/
srun rsync --progress ~/undertale_shared/models/item/$TOKENIZER $WORKSPACE/

python -m undertale.evaluation.basic_eval \
    --model $MODEL_LOC \
    --tokenizer $WORKSPACE/$TOKENIZER \
    --dataset $WORKSPACE/$DATASET/