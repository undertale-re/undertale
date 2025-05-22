#!/bin/bash
#SBATCH --job-name=mlm-humanevalx
#SBATCH -p gaia
#SBATCH --nodes=2
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=16G
#SBATCH --output=train-logs/ddp-mlm-humanevalx-%j.out

source /etc/profile
module load anaconda/2023b
eval "$(conda shell.bash hook)"
conda activate ~/.conda/envs/undertale-venv/

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=8888
export GPUS_PER_NODE=${SLURM_GPUS_ON_NODE}
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export RANK=$SLURM_PROCID
export OMP_NUM_THREADS=1

NNODES=${SLURM_JOB_NUM_NODES}

srun python -m torch.distributed.run \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m undertale.models.item.ddp-pretrain-maskedlm \
  ~/datasets/humanevalx-tokenized \
  -t ~/datasets/item.tokenizer.json \
  -o pretrain-maskedlm \
  -l debug
