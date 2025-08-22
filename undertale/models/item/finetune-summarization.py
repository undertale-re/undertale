import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl


from undertale.datasets.base import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain the model on a masked language modeling dataset",
    )
    
    
    parser.add_argument(
        "-e", "--end2end", default=True, type=bool, help="whether to train from assembly codo embeddings "
    )
    

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "dataset",
        help="dataset on which to train the model (format: `{module.path}:{DatasetClass}`)",
    )
    parser.add_argument("output", help="output model directory")

    parser.add_argument(
        "-c",
        "--checkpoint",
        help="trained model checkpoint from which to resume training",
    )
    
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")
    parser.add_argument(
        "-a", "--accelerator", default="auto", help="accelerator to use"
    )
    parser.add_argument(
        "-d",
        "--devices",
        default=1,
        type=int,
        help="number of accelerator devices to use (per node)",
    )
    parser.add_argument(
        "-n", "--nodes", default=1, type=int, help="number of nodes to use"
    )
    parser.add_argument("-v", "--version", help="training run version name")
    parser.add_argument(
        "--validation",
        action="store_true",
        help="whether to output validation examples",
    )

      
    arguments = parser.parse_args()



data=Dataset.load("/home/gridsan/AN31700/undertale_shared/datasets/xlcost-compiled-disassembled-parquet")
print(data.column_names)
print(data['disassembly'][0])
class SummarizeTrainer(pl.LightningModule):
    def __init__(self,model, prefix_length, total_steps,lr= 2e-5, warmup_steps=5000,end2end=True):
        
        self.model=model
        self.prompt_length=prompt_length
        self.lr=lr
        self.warmup_steps=warmup_steps
        self.total_steps=total_steps
        self.device=model.device
        self.end2end=True
    
    def forward(self,text,encoder_embedding,mask=None,labels=None):
        if end2end:
            embed_assembly(self,text,assembly_mask=None)
        return self.model(text,encoder_embedding,mask,labels)
        
    def training_step(self, batch, batch_idx):
        tokens, mask,disassembly_info = batch
        tokens, mask = tokens.to(self.device),mask.to(self.device)
        prefix = prefix.to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            self.model.embed_assembly(disassembly_info,assembly_mask=None):
        outputs = self(tokens, prefix,mask)
        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        
        self.log("train_loss", loss)
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        
        config_optim={
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1},
            }
        return config_optim 
        
                 