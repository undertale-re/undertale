python -m undertale.datasets.scripts.tokenize \
    ~/undertale_shared/datasets/nixpkgs-disassembled-rizin-pretokenized/ \
    ~/undertale_shared/datasets/nixpkgs-disassembled-rizin-pretraining/ \
    -t ~/undertale_shared/models/item/item.tokenizer.nixpkgs-disassembled-rizin.json \
    -w pretraining \
    -e slurm \
    -p 128
