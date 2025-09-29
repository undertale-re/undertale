#!/bin/sh
# finetune the existing model from checkpoint with pairwise dataset
#
# $1 is "local" if you are running on a Linux VM or box and "slurm" if
# you are working on the grid.
#
# $2 is path to the base pairwise dataset 
#
# This means, in fact that we have
#
# $3 item tokenizer json file
#
# $4 existing checkpoint model to be finetuned
#
# $5 output for the finetune
#
# $6 force gpu or cpu
#
# Both output documents containing "input_ids" in which strings tokens
# have been replaced by numbers, and "attention_maps" which are
# mysterious.
#
#

if ! [[ "$1" =~ ^(slurm|local)$ ]];
then
	echo "Usage: finetune.sh [slurm|local] path_to_base_rzpairs path_to_item_tokenizer fullpath_model_checkpoint path_to_output_finetune"
    exit 1
fi




if [ "$1" == "local" ]; then
    python -m undertale.models.item.finetune_embedding \
           $2/ \
           -t $3 \
           -m $4 \
           -o $5 \
	   -a $6 \ 
           -e local 
else
       python -m undertale.models.item.finetune_embedding \
           $2/ \
           -t $3 \
           -m $4 \
           -o $5 \
           -a $6 \ 
           -e slurm \
           -p 128
fi
