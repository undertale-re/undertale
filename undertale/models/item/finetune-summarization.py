import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import transformers


from types import SimpleNamespace

from  .summarization_dataset import SummarizerDataset, CustomCollator
from .model import TransformerEncoderForSequenceSummarization
from ... import logging as undertale_logging

from datasets import load_dataset
from undertale.datasets.base import Dataset
from . import tokenizer

from torch.utils.data import DataLoader

from lightning import Trainer,LightningModule
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import time

class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    
    
# class ValidationCallback(Callback):
#     def __init__(self, dataloader, tok):
#         super().__init__()
#         self.dataloader = dataloader
#         self.tokenizer = tok

#     def on_validation_end(self, trainer, pl_module):
#         for batch in self.dataloader:
#             input_ids = batch.input_ids.to(pl_module.device)
#             attention_mask = batch.attention_mask.to(pl_module.device)
#             output = pl_module(input_ids, attention_mask)
#             filled = torch.where(
#                 input_ids == self.tokenizer.token_to_id(tokenizer.TOKEN_MASK),
#                 torch.argmax(output, dim=-1),
#                 input_ids,
#             )
#             input_seq = (
#                 self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
#                 .replace("[NEXT]", "\n")
#                 .replace("[PAD]", "")
#                 .strip()
#             )
#             predicted = self.tokenizer.decode(
#                 filled[0].tolist(), skip_special_tokens=False
#             )
#             predicted = (
#                 predicted.replace(tokenizer.TOKEN_PAD, "")
#                 .replace("[NEXT]", "\n")
#                 .strip()
#             )
#             if isinstance(pl_module.logger.experiment, SummaryWriter):
#                 pl_module.logger.experiment.add_text(
#                     "mask prediction", f"input: {input_seq}\n\noutput:{predicted}"
#                )
    
class SummarizeModel(LightningModule,torch.nn.Module):
    def __init__(self, model, prefix_length,lr= 2e-5, warmup_steps=5000,end2end=True):
        super().__init__()
        
        self.model=model
        self.prefix_length=prefix_length
        self.lr=lr
       
        self.warmup_steps=warmup_steps
        self.end2end=True
        

    
    def forward(self,text,encoder_embedding,mask=None,labels=None):

        return self.model(text,encoder_embedding,mask,labels)
        
    def training_step(self, batch, batch_idx):

        tokens, mask,dissassembly_tokens,dissassembly_mask=batch['tokens'], batch['mask'],batch['disassembly_tokens'],batch['disassembly_mask'] 
        #tokens, mask = tokens.to(self.device),mask.to(self.device)
        # disassembly_info = disassembly_info.to(self.device, dtype=torch.float32)
        

        if self.end2end:
            with torch.no_grad():

                prefix=self.model.embed_assembly(dissassembly_tokens,dissassembly_mask)

        else:
            prefix=disassembly_info
          
        outputs = self(tokens, prefix,mask)

        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        tokens, mask,dissassembly_tokens,dissassembly_mask=batch['tokens'], batch['mask'],batch['disassembly_tokens'],batch['disassembly_mask'] 
        # tokens, mask = tokens.to(self.device),mask.to(self.device)
        # disassembly_info = disassembly_info.to(self.device, dtype=torch.float32)
        
        
        if self.end2end:
            with torch.no_grad():
                prefix=self.model.embed_assembly(dissassembly_tokens,dissassembly_mask)
        else:
            prefix=disassembly_info
            
        outputs = self(tokens, prefix,mask)
        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        
        self.log("val_loss", loss)
        
        return loss
    
    
    def configure_optimizers(self):
        
        total_steps = self.trainer.estimated_stepping_batches
        
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        
        config_optim={
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1},
            }
        return config_optim 
        
                 

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        description="pretrain the model on a masked language modeling dataset",
    )
    
   
    
    #model info
    parser.add_argument('--gpt2path', type=str)
    
    parser.add_argument(
        "--assembly_checkpoint",type=str,help="trained model checkpoint from which to resume training")
    
    parser.add_argument(
        "-c",
        "--summarizer_checkpoint",
        help="trained model checkpoint from which to resume training",
    )
    
    parser.add_argument('--model_type', type=str, default="transformer",
                        help="model for connector")
    parser.add_argument('--prefix_length_const', type=int,default=10,help="length for additional prefix constant tokens")
    parser.add_argument('--prefix_length_assembly', type=int, default=10, help='break down assembly output into sequence length')
    parser.add_argument('--num_layers', type=int, default=8, help='number layers for transformer')
    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained assembly tokenizer file"
    )
    parser.add_argument('--tokenizer_size', type=int, default=512)
    parser.add_argument(
        "-e", "--end2end", dest='end2end', action='store_true',help="whether to train from assembly codo embeddings "
    )
    parser.add_argument('--tune_llm', dest='tune_llm', action='store_true')
    
    #dataset info
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--token_batchsize', type=int, default=1024)
    parser.add_argument('--dataset', type=str,help="dataset on which to train the model")
    parser.add_argument("--seed", type=int, default=42, help="seed to split dataset")
    parser.add_argument("--test_size", type=float, default=0.1, help="ratio size of test set. remaining size is train set")
    
    
    
    #training info
    parser.add_argument("--output", help="output model directory")

    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-warmup", "--warmup_steps", type=int, default=5000, help="number of warmup steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument(
        "-a", "--accelerator", default="auto", help="accelerator to use"
    )
    parser.add_argument("-num_epochs", type=int, default=10, help="number epochs to train model")
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
    
    

      
    args = parser.parse_args()
    
    
    undertale_logging.setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    
    #set up dataloaders
    if args.end2end:
        dataset=Dataset.load(args.dataset)

    elif os.path.exists(args.dataset):
            dataset = load_dataset("parquet", data_files=args.dataset)
    else:
        raise FileNotFoundError(f"File not found: {args.dataset}")
    
    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    
    train_dataset = SummarizerDataset(dataset=split_dataset["train"], prefix_length=args.prefix_length_const, gpt2path=args.gpt2path,
                              normalize_prefix=args.normalize_prefix,end2end=args.end2end,
                              token_batchsize=args.token_batchsize)
    
    val_dataset = SummarizerDataset(dataset=split_dataset["test"], prefix_length=args.prefix_length_const, gpt2path=args.gpt2path,
                              normalize_prefix=args.normalize_prefix,end2end=args.end2end,
                              token_batchsize=args.token_batchsize)
    
    
    
    #assembly_tokenizer = tokenizer.load(args.tokenizer)
    collator = CustomCollator(args,train_dataset.max_seq_len,device)
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collator
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator
    )
    
    
    #set up model
    connector_config={"model_type":args.model_type,
                      "prefix_size":768,
                      "prefix_length_const":args.prefix_length_const,
                     "prefix_length_assembly":args.prefix_length_assembly,
                     "num_layers":args.num_layers}
    
    connector_config=SimpleNamespace(**connector_config)
    
    model=TransformerEncoderForSequenceSummarization(args.assembly_checkpoint,connector_config,
                                                     args.gpt2path,args.end2end,args.tune_llm)
    
    
    output = os.path.abspath(os.path.expanduser(args.output))

    
    
    progress = ProgressBar(leave=True)
    checkpoint = ModelCheckpoint(
        filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}",
        save_top_k=-1,
    )
    stop = EarlyStopping(monitor="val_loss", mode="max", patience=5, min_delta=0.001)
    logger = TensorBoardLogger(
        save_dir=os.path.dirname(output),
        name=os.path.basename(output),
        version=args.version,
    )

    if args.validation:
        pass
        random_sampler = RandomSampler(dataset["test"], num_samples=5)
        random_validation = DataLoader(
            dataset["test"],
            sampler=random_sampler,
            batch_size=1,
            collate_fn=collator,
            num_workers=1,
        )
        validation_check = ValidationCallback(random_validation, tok)
        callbacks = [progress, checkpoint, stop, validation_check]
    else:
        callbacks = [progress, checkpoint, stop]
    
    summarize_model=SummarizeModel(model, args.prefix_length_const,lr=args.lr, warmup_steps=args.warmup_steps,end2end=args.end2end)

    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        max_epochs=args.num_epochs,
        # Testing
        # log_every_n_steps=1,
        # limit_train_batches=2,
        # limit_val_batches=2,
    )
    trainer.fit(
        summarize_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.summarizer_checkpoint,
    )


