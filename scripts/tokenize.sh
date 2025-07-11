#!/bin/bash

# tokenize dataset which has already been pre-tokenized
#
# $1 is "local" if you are running on a Linux VM or box and "slurm" if
# you are working on the grid.
#
# $2 is path to the base dataset with disassembly that you want to tokenize
#    e.g., ~//undertale_shared/datasets/assemblage-windows-disassembled-rz
#
# This means, in fact that we have
#   input is $2-pretokenized
#   output is $2-pretraining
#
# $3 is path to tokenizer file,
#    e.g. ~/undertale_shared/models/item/item.tokenizer.nixpkgs-disassembled-rizin.json
#
# $4 is either "pretraining" or "pretraining_equiv_classes"
#
# Both output documents containing "input_ids" in which strings tokens
# have been replaced by numbers, and "attention_maps" which are
# mysterious.
#
# pretraining_equiv_classes, additionally, outputs an "equiv_class"
# field, carried forward from the pretokenized version. So it better
# be there.  Use this to generate a tokenized dataset for generating
# pairs for contrastive fine-tuning of embeddings.
#

if ! [[ "$1" =~ ^(slurm|local)$ ]];
then
    echo "Usage: tokenize.sh [slurm|local] path_to_base_dataset path_to_tokenizer [pretraining|pretraining_equiv_classes]"
    exit 1
fi

if ! [[ "$4" =~ ^(pretraining|pretraining_equiv_classes)$ ]];
then
    echo "Usage: tokenize.sh [slurm|local] path_to_base_dataset path_to_tokenizer [pretraining|pretraining_equiv_classes]"
    error
fi


if [ "$1" == "local" ]; then
    python -m undertale.datasets.scripts.tokenize \
           $2-pretokenized/ \
           $2-pretraining/ \
           -t $3 \
           -q \
           -w $4 \
           -e local 
else
    python -m undertale.datasets.scripts.tokenize \
           $2-pretokenized/ \
           $2-pretraining/ \
           -t $3 \
           -q \
           -w $4 \
           -e slurm \
           -p 128
fi

#python -m undertale.datasets.scripts.tokenize \
#    ~/undertale_shared/datasets/nixpkgs-disassembled-rizin-pretokenized/ \
#    ~/undertale_shared/datasets/nixpkgs-disassembled-rizin-pretraining/ \
#    -t ~/undertale_shared/models/item/item.tokenizer.nixpkgs-disassembled-rizin.json \
#    -w pretraining \
#    -e slurm \
#    -p 128
