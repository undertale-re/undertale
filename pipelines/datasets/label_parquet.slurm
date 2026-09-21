#!/bin/bash
#SBATCH --job-name=label-parquet
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --distribution=nopack
#SBATCH --output=label-parquet-%A_%a.out

# Original parquet shards -> labeled parquet shards, in one pass. The job is
# client-only: every array task queries one vLLM server that is already running
# elsewhere. Start that server separately, then point this job at it:
#
#   ENDPOINT=http://{node-name}:{vllm-port} sbatch label_parquet.slurm
#
# Submit from this directory. The job runs the `label_parquet.py` beside this
# file and falls back to the `function_taxonomy.json` beside it too.
#
# Work is split by shard file, not by row, so each task writes whole parquet
# files and shards already present are skipped. A job that runs out of time
# resumes by resubmitting it unchanged -- no listing to rebuild first.
#
# The array size is how many clients share the one server, not how the corpus is
# divided; 965 shards across 8 tasks is ~121 each. Raise it (and --concurrency)
# only as far as the server keeps up.

DATASET=${DATASET:-nixpkgs-legacy_10percent}
INPUTS=${INPUTS:-$SHARE/datasets/$DATASET/split-testing}
DESTINATION=${DESTINATION:-$SHARE/datasets/$DATASET/split-testing-labeled}
ENDPOINT=${ENDPOINT:-http://localhost:8000}
SHARDS=$SLURM_ARRAY_TASK_COUNT

# Accept the endpoint with or without the trailing OpenAI-compatible /v1 path.
ENDPOINT=${ENDPOINT%/}
[[ $ENDPOINT == */v1 ]] || ENDPOINT=$ENDPOINT/v1

# Fail fast and with a clear message if the server is not reachable, rather than
# letting every request in the shard time out one by one.
if ! curl -sf --noproxy '*' "$ENDPOINT/models" > /dev/null; then
    echo "no vLLM server reachable at $ENDPOINT"
    exit 1
fi

# Left unset, label_parquet.py asks the server what it is serving, so the model
# recorded in the output cannot drift away from the one that answered. TAXONOMY
# likewise only has to be set to override the file shipped alongside.
OPTIONS=()
if [[ -n $MODEL ]]; then
    OPTIONS+=(--model "$MODEL")
fi
if [[ -n $TAXONOMY ]]; then
    OPTIONS+=(--taxonomy "$TAXONOMY")
fi

echo "shard $SLURM_ARRAY_TASK_ID/$SHARDS -> $ENDPOINT ${MODEL:-(server default)}"

python label_parquet.py \
    "$INPUTS" \
    --destination "$DESTINATION" \
    --endpoint "$ENDPOINT" \
    "${OPTIONS[@]}" \
    --shard $SLURM_ARRAY_TASK_ID \
    --shards $SHARDS \
    --concurrency 32
