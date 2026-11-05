#!/bin/bash
#SBATCH --job-name=label-parquet
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --distribution=nopack
#SBATCH --output=label-parquet-%A_%a.out

# Original parquet shards -> labeled parquet shards, in one pass. The job is
# client-only -- it only reads parquet and talks HTTP to a vLLM server running
# elsewhere -- so it asks for little: 2 CPUs and 4G, not the GPU-job defaults
# (--cpus-per-task=12, --mem=0) it was copied from. A smaller request schedules
# faster. Raise them only if a shard's parquet row group genuinely needs more.
#
# Every array task queries the one vLLM server. Start that server separately
# (we have been serving Qwen3-Coder-30B-A3B-Instruct), then point this job at it:
#   ENDPOINT=http://{node-name}:{vllm-port} sbatch label_parquet.slurm
#
# Submit from this directory. The job runs the `label_parquet.py` beside this
# file and falls back to the `function_taxonomy.json` beside it too.
#
# Work is split by shard file, not by row, so each task writes whole parquet
# files and shards already present are skipped. A job that runs out of time
# resumes by resubmitting it unchanged -- no listing to rebuild first.
#
# If the vLLM server dies or stalls mid-run, label_parquet.py aborts any shard
# whose connection/HTTP error rate crosses --max-error-rate (MAX_ERROR_RATE,
# default 0.25): that shard is left unwritten rather than renamed into place
# near-empty, and the task exits non-zero so Slurm marks it FAILED instead of
# silently losing data. Resubmit the job unchanged once the server is back and
# the aborted shards are retried, while the ones already finished are skipped.
#
# The array size is how many clients share the one server, not how the corpus is
# divided; 965 shards across 8 tasks is ~121 each. Raise it (and --concurrency)
# only as far as the server keeps up.

DATASET=${DATASET:-nixpkgs-legacy_10percent}
INPUTS=${INPUTS:-$SHARE/datasets/$DATASET/split-validation}
DESTINATION=${DESTINATION:-$SHARE/datasets/$DATASET/split-validation-labeled}
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
if [[ -n $MAX_ERROR_RATE ]]; then
    OPTIONS+=(--max-error-rate "$MAX_ERROR_RATE")
fi

echo "shard $SLURM_ARRAY_TASK_ID/$SHARDS -> $ENDPOINT ${MODEL:-(server default)}"

python label_parquet.py \
    "$INPUTS" \
    --destination "$DESTINATION" \
    --endpoint "$ENDPOINT" \
    "${OPTIONS[@]}" \
    --shard $SLURM_ARRAY_TASK_ID \
    --shards $SHARDS \
    --drop-unlabeled \
    --concurrency 32
status=$?

# A non-zero exit means one or more shards were aborted (usually a dead or
# stalled server), so those shards were left unwritten. Surface it clearly and
# propagate it, so the task shows up as FAILED and a plain resubmit retries them.
if (( status != 0 )); then
    echo "shard $SLURM_ARRAY_TASK_ID/$SHARDS: exit $status -- shard(s) aborted," \
         "left unwritten; resubmit once the server is healthy to retry them"
fi
exit $status
