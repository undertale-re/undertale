import argparse

from lightning.pytorch.utilities.model_summary import ModelSummary

from . import tokenizer
from .model import TransformerEncoderForSequenceSimilarity

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="finetune the model on a pairwise embedding dataset",
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )

    parser.add_argument("-f", "--checkpoint", required=True, help="checkpointmodel")

    arguments = parser.parse_args()

    tok = tokenizer.load(arguments.tokenizer, sequence_length=512)
    modelpath = (
        arguments.checkpoint
    )  # "/panfs/g52-panfs/scratch/as28456/assemblage-windows-disassembled-finetune-embedding/version_1.0/checkpoints/epoch=1-train_loss=0.19-val_loss=0.20.ckpt"
    model = TransformerEncoderForSequenceSimilarity.load_from_checkpoint(modelpath)
    model_state_dict = model.state_dict()

    summary = ModelSummary(model, max_depth=-1)  # Use -1 to show all modules
    print(summary)

    for param_name, param_tensor in model_state_dict.items():
        # print("Param Features",param_name)
        if param_name.startswith("encoder.embedding.token") or param_name.startswith(
            "encoder.layers.11.ff.linear2"
        ):
            print(f"{param_name}\t{param_tensor}")
