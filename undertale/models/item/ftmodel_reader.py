from .model import Defaults, TransformerEncoderForSequenceSimilarity,TransformerEncoderForMaskedLM
from lightning.pytorch.utilities.model_summary import ModelSummary
from . import tokenizer
import torch
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="finetune the model on a pairwise embedding dataset",
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )

    parser.add_argument(
        "-f", "--checkpoint", required=True, help="checkpointmodel"
    )

    arguments = parser.parse_args()

    tok = tokenizer.load(arguments.tokenizer, sequence_length=512)
    ''' 
    model = TransformerEncoderForSequenceSimilarity(
        depth=Defaults.depth,
        hidden_dimensions=Defaults.hidden_dimensions,
        vocab_size=tok.get_vocab_size(),
        input_size=Defaults.input_size,
        heads=Defaults.heads,
        intermediate_dimensions=Defaults.intermediate_dimensions,
        dropout=Defaults.dropout,
        eps=Defaults.eps,
        lr=Defaults.lr, 
        warmup=Defaults.warmup,
        #embedding_size=128, #ASK TODO REVISIT
        #embedding_dropout_prob=Defaults.dropout
    )
    '''
    modelpath = arguments.checkpoint #"/panfs/g52-panfs/scratch/as28456/assemblage-windows-disassembled-finetune-embedding/version_1.0/checkpoints/epoch=1-train_loss=0.19-val_loss=0.20.ckpt"
    model = TransformerEncoderForSequenceSimilarity.load_from_checkpoint(modelpath)
    #model.from_pretrained(modelpath, local_files_only=True)
    model_state_dict = model.state_dict() 
    #model = torch.load(modelpath) 
    #model_state_dict = model['state_dict']

    summary = ModelSummary(model, max_depth=-1) # Use -1 to show all modules
    print(summary)

    for param_name, param_tensor in model_state_dict.items():
        #print("Param Features",param_name)
        if (param_name.startswith('encoder.embedding.token') or param_name.startswith('encoder.layers.11.ff.linear2')):
            print(f"{param_name}\t{param_tensor}")
