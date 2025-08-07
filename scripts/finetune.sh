#!/bin/sh
python -m undertale.models.item.finetune-embedding /panfs/g52-panfs/scratch/as28456/assemblage-windows-disassembled-rz-pairs/ -t item.tokenizer.json -m /panfs/g52-panfs/scratch/as28456/nixpkgs_models/nixpkgs-small-pytorch/checkpoints/'epoch=8-train_loss=0.34-valid_f1=0.92.ckpt' -o /panfs/g52-panfs/scratch/as28456/assemblage-windows-disassembled-finetune-embedding -e 1

