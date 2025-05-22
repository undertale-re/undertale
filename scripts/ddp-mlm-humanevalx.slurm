#!/bin/bash
#SBATCH --job-name=mlm-humanevalx
#SBATCH --exclusive 
#SBATCH -p gaia
#SBATCH --nodes=4
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=16G
#SBATCH --output=train-logs/ddp-mlm-humanevalx-%j.out

source /etc/profile
module load anaconda/2023b
source activate ~/.conda/envs/undertale-venv

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=8888
export GPUS_PER_NODE=2
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun torchrun \
  --nproc_per_node=2 \
  --nnodes=4 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m undertale.models.item.ddp-pretrain-maskedlm \
  -i ~/datasets/humaneval-x \
  -t item.tokenizer.json \
  -o pretrain-maskedlm
