import argparse
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from datasets import load_dataset
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

# from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from rouge_score import rouge_scorer
from bert_score import score as bert_score

from undertale.datasets.base import Dataset

from ... import logging as undertale_logging

# from . import tokenizer
from .model import TransformerEncoderForSequenceSummarization
from .summarization_dataset import CustomCollator, SummarizerDataset


def dataset_size_type(x):
    x = int(x)
    if x == 0:
        raise argparse.ArgumentTypeError("dataset_size cannot be 0. Use -1 for full dataset or a positive integer.")
    return x

class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class ValidationCallback(Callback):
    def __init__(
        self,
        dataloader,
        run_on_val_end=True, 
        run_on_fit_end=False, 
        tag=None,
        end2end=False,
        args=None,
    ):
        super().__init__()
        
        
        self.bertscore_model_path=args.bertscore_model_path
        self.dataloader = dataloader
        self.save_dir = args.generated_output_paths
        self.end2end = end2end
        self.run_on_val_end=run_on_val_end
        self.run_on_fit_end=run_on_fit_end
        self.tag=tag
        
        # ---- generation config (EXPLICIT) ----
        self.beam = args.beam
        self.num_beams = args.num_beams
        self.temperature=args.temperature
        self.max_new_tokens = args.max_new_tokens


        self.do_sample = False

        os.makedirs(self.save_dir, exist_ok=True)
        
        
    def _run_validation(self,trainer,pl_module):
        if not trainer.is_global_zero:
            return

        outputs = []
        was_training = pl_module.training
        pl_module.eval()
        device = pl_module.device
        
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_l_f1_sum = 0.0
        n_samples = 0

        bert_refs = []
        bert_cands = []

        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):


                tokens = batch["tokens"].to(device)
                mask = batch["mask"].to(device)
                dis_tokens = batch["disassembly_tokens"].to(device)
                dis_mask = batch["disassembly_mask"].to(device)

                # ---------------- encoder path (UNCHANGED) ----------------
                if self.end2end:
                    encoder_embedding = pl_module.model.embed_assembly(
                        dis_tokens, dis_mask
                    )
                    encoder_embedding = encoder_embedding.mean(dim=1)
                else:
                    encoder_embedding = dis_tokens
                    if encoder_embedding.dim() == 3:
                        encoder_embedding = encoder_embedding.mean(dim=1)

                prefixes = pl_module.model.connector(encoder_embedding).view(
                    -1,
                    pl_module.model.prefix_length_const,
                    pl_module.model.llm_embedding_size,
                )

                # ---------------- generation ----------------
                text = pl_module.model.generate(
                    embed=prefixes,
                    entry_length=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    beam=self.beam,
                    num_beams=self.num_beams,
                )
                prefix_len = pl_module.model.prefix_length_const
                token_mask = mask[:, prefix_len:]
                caption = pl_module.model.tokenizer.decode(
                    tokens[0][token_mask[0] == 1], skip_special_tokens=True
                )
                outputs.append([caption, 
                                text])
                caption=caption.replace("_", " ")
                text=text.replace("_"," ")
                rouge_score = rouge.score(caption, text)["rougeL"].fmeasure
                rouge_l_f1_sum += rouge_score

                # BERTScore (accumulate only)
                bert_refs.append(caption)
                bert_cands.append(text)

                n_samples += 1

        if was_training:
            pl_module.train()
            
            
        rouge_l_f1 = rouge_l_f1_sum / max(1, n_samples)

        P, R, F1 = bert_score(
            bert_cands,
            bert_refs,
            lang="en",
            num_layers=24,
            model_type=self.bertscore_model_path,
            device=str(device),
            verbose=False,
        )
        bert_f1 = F1.mean().item()

        trainer.print(
            f"[val metrics] ROUGE-L(F1)={rouge_l_f1:.4f} | "
            f"BERTScore(F1)={bert_f1:.4f}"
        )
        
        if self.tag==None:
            path = os.path.join(
                self.save_dir, f"epoch_{trainer.current_epoch}.txt"
            )
        else:
            path = os.path.join(
                self.save_dir, f"{self.tag}_epoch_{trainer.current_epoch}.txt"
            )
        self.log("rouge_l_f1", rouge_l_f1)
        self.log("bert_f1", bert_f1)
        with open(path, "w") as f:
            
            f.write(
            f"ROUGE-L(F1): {rouge_l_f1:.6f}\n"
            f"BERTScore(F1): {bert_f1:.6f}\n"
            f"NUM_SAMPLES: {n_samples}\n"
            "=================\n\n"
            )
            
            for cap, pred in outputs:
                f.write(f"GROUND TRUTH CAPTION:\n{cap}\n\n")
                f.write(f"PREDICTED CAPTION:\n{pred}\n\n")
                f.write("_________________\n")

        trainer.print(f"Saved validation outputs to {path}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.run_on_val_end:
            self._run_validation(trainer,pl_module)
        
    def on_train_end(self, trainer, pl_module):
        if self.run_on_fit_end:
            self._run_validation(trainer, pl_module)
        



class SummarizeModel(LightningModule, torch.nn.Module):
    def __init__(self, model, prefix_length, lr=2e-5, warmup_steps=5000, end2end=True):
        super().__init__()

        self.model = model
        self.prefix_length = prefix_length
        self.lr = lr

        self.warmup_steps = warmup_steps
        self.end2end = end2end

    def forward(self, text, encoder_embedding, mask=None, labels=None):

        return self.model(text, encoder_embedding, mask, labels)

    def training_step(self, batch, batch_idx):
        
        tokens, mask, dissassembly_tokens, dissassembly_mask = (
            batch["tokens"],
            batch["mask"],
            batch["disassembly_tokens"],
            batch["disassembly_mask"],)
        
        if self.end2end:
            with torch.no_grad():

                prefix = self.model.embed_assembly(
                    dissassembly_tokens, dissassembly_mask
                )
        else:
            prefix = dissassembly_tokens

        outputs = self(tokens, prefix, mask)

        logits = outputs.logits[:, self.prefix_length - 1 : -1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0
        )
        
        self.log("train_loss", loss)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):

        tokens, mask, dissassembly_tokens, dissassembly_mask = (
            batch["tokens"],
            batch["mask"],
            batch["disassembly_tokens"],
            batch["disassembly_mask"],
        )
        # tokens, mask = tokens.to(self.device),mask.to(self.device)
        # disassembly_info = disassembly_info.to(self.device, dtype=torch.float32)

        if self.end2end:
            with torch.no_grad():
                prefix = self.model.embed_assembly(
                    dissassembly_tokens, dissassembly_mask
                )
        else:
            prefix = dissassembly_tokens

        outputs = self(tokens, prefix, mask)
        logits = outputs.logits[:, self.prefix_length - 1 : -1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0
        )

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches

        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

        warmup_steps = 5000
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=1000000,
        )

        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.warmup_steps,
        #    num_training_steps=total_steps,
        #)

        config_optim = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return config_optim


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="pretrain the model on a masked language modeling dataset",
    )

    # model info
    parser.add_argument("--gpt2path", type=str)

    parser.add_argument(
        "--assembly_checkpoint",
        type=str,
        help="trained model checkpoint from which to resume training",
    )

    parser.add_argument(
        "-c",
        "--summarizer_checkpoint",
        help="trained model checkpoint from which to resume training",
    )

    parser.add_argument(
        "--model_type", type=str, default="transformer", help="model for connector"
    )
    parser.add_argument(
        "--beam",
        action="store_true",
        help="whether to use beam search for generation of text",
    )
    
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="if beam is toggled, number of beams to use",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="max number tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for generation",
    )

    parser.add_argument(
        "--prefix_length_const",
        type=int,
        default=40, #i increased from 10 (without tuning with llm)
        help="length for additional prefix constant tokens",
    )
    parser.add_argument(
        "--prefix_length_assembly",
        type=int,
        default=40, #i increased from 10 (without tuning with llm)
        help="break down assembly output into sequence length",
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="number layers for transformer"
    )
    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained assembly tokenizer file"
    )
    parser.add_argument("--tokenizer_size", type=int, default=512)
    parser.add_argument(
        "-e",
        "--end2end",
        dest="end2end",
        action="store_true",
        help="whether to train from assembly codo embeddings ",
    )
    parser.add_argument("--tune_llm", dest="tune_llm", action="store_true")

    # dataset info
    parser.add_argument(
        "--normalize_prefix", dest="normalize_prefix", action="store_true"
    )
    parser.add_argument("--token_batchsize", type=int, default=1024)
    parser.add_argument(
        "--dataset", type=str, help="dataset on which to train the model"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed to split dataset")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="ratio size of test set. remaining size is train set",
    )
    parser.add_argument(
    "--dataset_size",
    type=dataset_size_type,
    default=-1,
    help="subsample dataset. -1 means use entire dataset. will throw error is choose 0"
    )
    

    # training info
    parser.add_argument("--output", help="output model directory")

    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "-warmup",
        "--warmup_steps",
        type=int,
        default=1, #used to be 50000
        help="number of warmup steps",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate") #used to be 2e-5
    parser.add_argument(
        "-a", "--accelerator", default="auto", help="accelerator to use"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="number epochs to train model"
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
    parser.add_argument(
        "--generated_output_paths",
        default="./validation_outputs",
        type=str,
        help="where to output validation examples",
    )
    parser.add_argument(
        "--bertscore_model_path",
        type=str,
        help="path for bert score model",
    )
   
    args = parser.parse_args()
    
    #cache_root= args.bertscore_model_path
    
    # os.environ["HF_HUB_CACHE"] = cache_root
    # os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # # optional but helps with older stacks:
    # os.environ["HF_HOME"] = cache_root
    # os.environ["TRANSFORMERS_CACHE"] = cache_root
    # # optional: for older huggingface_hub versions:
    # os.environ["HUGGINGFACE_HUB_CACHE"] = cache_root
    
    

    undertale_logging.setup_logging()

    # set up dataloaders
    if args.end2end:
        dataset = Dataset.load(args.dataset)

    elif os.path.exists(args.dataset):
        dataset = load_dataset("parquet", data_files=args.dataset)
    else:
        raise FileNotFoundError(f"File not found: {args.dataset}")
        
        
    cols = dataset.column_names
    if "function_name" in cols and "summary" not in cols:
        dataset = dataset.rename_column("function_name", "summary")
    
    dataset = dataset.select_columns(["disassembly", "summary"])
    
    if not args.dataset_size==-1:
        dataset=dataset.select(range(args.dataset_size))
    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    train_dataset = SummarizerDataset(
        dataset=split_dataset["train"],
        prefix_length=args.prefix_length_const,
        gpt2path=args.gpt2path,
        normalize_prefix=args.normalize_prefix,
        end2end=args.end2end,
        token_batchsize=args.token_batchsize,
    )

    val_dataset = SummarizerDataset(
        dataset=split_dataset["test"],
        prefix_length=args.prefix_length_const,
        gpt2path=args.gpt2path,
        normalize_prefix=args.normalize_prefix,
        end2end=args.end2end,
        token_batchsize=args.token_batchsize,
    )

    # assembly_tokenizer = tokenizer.load(args.tokenizer)
    collator = CustomCollator(args, train_dataset.max_seq_len, train_dataset.pad_id)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collator,num_workers=8, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collator,num_workers=8, drop_last=True
    )

    # set up model
    connector_config = {
        "model_type": args.model_type,
        "prefix_size": 768,
        "prefix_length_const": args.prefix_length_const,
        "prefix_length_assembly": args.prefix_length_assembly,
        "num_layers": args.num_layers,
    }

    connector_config = SimpleNamespace(**connector_config)

    model = TransformerEncoderForSequenceSummarization(
        args.assembly_checkpoint,
        connector_config,
        args.gpt2path,
        args.end2end,
        args.tune_llm,
    )

    output = os.path.abspath(os.path.expanduser(args.output))

    progress = ProgressBar(leave=True)
    checkpoint = ModelCheckpoint(
        filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}",
        save_top_k=-1,
    )
    stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.001)
    logger = TensorBoardLogger(
        save_dir=os.path.dirname(output),
        name=os.path.basename(output),
        version=args.version,
    )

    if args.validation:

        random_sampler = RandomSampler(val_dataset, num_samples=5)
        random_validation = DataLoader(
            val_dataset,
            sampler=random_sampler,
            batch_size=1,
            collate_fn=collator,
            num_workers=8
        )

        final_validation=DataLoader(
            val_dataset,
            batch_size=1,
            collate_fn=collator,
            num_workers=8
        )
        
        final_validation_check=ValidationCallback(
                                    final_validation, 
                                    end2end=args.end2end,
                                    tag="final_full_val",
                                    run_on_val_end=False, 
                                    run_on_fit_end=True,
                                    args=args)
        
        validation_check = ValidationCallback(
                                    random_validation, 
                                    end2end=args.end2end,
                                    tag=None,
                                    run_on_val_end=True, 
                                    run_on_fit_end=False,
                                    args=args)
        
        
        callbacks = [progress, checkpoint, stop, validation_check,final_validation_check]

    else:
        callbacks = [progress, checkpoint, stop]

    summarize_model = SummarizeModel(
        model,
        args.prefix_length_const,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        end2end=args.end2end,
    )

    sample = val_dataloader.dataset[0]

    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        max_steps=1000000,
        val_check_interval=2000,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        #max_epochs=args.num_epochs,
        # Testing
        # log_every_n_steps=1,
        # limit_train_batches=2,
        # limit_val_batches=2,
    )
    
    # trainer = Trainer(
    #     strategy="ddp_find_unused_parameters_true",
    #     max_epochs=1,
    #     callbacks=callbacks,
    #     limit_train_batches=1,
    #     limit_val_batches=1,
    #     num_sanity_val_steps=0,
    #     # optionally:
    #     devices=args.devices,
    #     logger=False,
    # )
    trainer.fit(
        summarize_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.summarizer_checkpoint,
    )
    
    
    
