from .model import Defaults, TransformerEncoderForSequenceSimilarity,TransformerEncoderForMaskedLM
from lightning.pytorch.utilities.model_summary import ModelSummary
from . import tokenizer
import torch
if __name__ == "__main__":

    tok = tokenizer.load("item.tokenizer.json", sequence_length=512)
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
    modelpath = "/panfs/g52-panfs/scratch/as28456/assemblage-windows-disassembled-finetune-embedding/version_1.0/checkpoints/epoch=6-train_loss=0.14-val_loss=0.20.ckpt"
    model = TransformerEncoderForSequenceSimilarity.load_from_checkpoint(modelpath)
    #model.from_pretrained(modelpath, local_files_only=True)
    model_state_dict = model.state_dict() 
    #model = torch.load(modelpath) 
    #model_state_dict = model['state_dict']
    for param_name, param_tensor in model_state_dict.items():
        if (param_name =='encoder.embedding.token.weight'):
            print(f"{param_name}\t{param_tensor}")

    summary = ModelSummary(model, max_depth=-1) # Use -1 to show all modules
    print(summary)