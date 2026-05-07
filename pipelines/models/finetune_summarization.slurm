#!/bin/bash
#SBATCH --job-name=finetune-summarization
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --distribution=nopack
#SBATCH --output=summarizer_train_logs/finetune-summarization-%j
#SBATCH --time=UNLIMITED


LOG_DIR=summarizer_train_logs
mkdir -p "$LOG_DIR"

DATASET=nixpkgs
TOKENIZER=$SHARE/models/tokenizer/nixpkgs-legacy/latest
TRAINING=$SHARE/datasets/nixpkgs/pretokenized/20260212-132053
ASSEMBLY_CHECKPOINT=$SHARE/models/maskedlm/nixpkgs-legacy/latest
GPT2=$LOCAL/models/gpt2
BERTSCORE_MODEL=$LOCAL/models/roberta-large
OUTPUT=$LOCAL/output/function_summarization
GENERATED_OUTPUTS=$LOCAL/undertale/validation_outputs/no_beam
BATCH=46
EPOCHS=50
DATASET_SIZE=-1

export PYTHONUNBUFFERED=TRUE
export TOKENIZERS_PARALLELISM=false

srun python -m pipelines.models.finetune_summarization \
    --tokenizer $TOKENIZER 
    --output $OUTPUT \
    --gpt2path $GPT2 \
    --assembly_checkpoint $ASSEMBLY_CHECKPOINT \
    --dataset $TRAINING \
    --generated_output_paths $GENERATED_OUTPUTS \
    --end2end \
    --num_epochs $EPOCHS \
    --dataset_size $DATASET_SIZE \
    --validation \
    --tune_llm \
    --bertscore_model_path $BERTSCORE_MODEL \
    --batch_size $BATCH \
    --accelerator gpu \
    --nodes $SLURM_JOB_NUM_NODES \
    --devices $SLURM_NTASKS_PER_NODE \
    --version $SLURM_JOB_ID
