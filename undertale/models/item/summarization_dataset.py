from transformers import GPT2Tokenizer,DataCollatorWithPadding
import torch
import multiprocessing
import numpy as np
import os

from transformers import PreTrainedTokenizerFast,AutoTokenizer, AutoConfig
from tqdm import tqdm
from . import tokenizer

class SummarizerDataset(torch.utils.data.Dataset):
    
    
    def __init__(self, dataset, prefix_length, gpt2path,
                 normalize_prefix=False,end2end=True,token_batchsize=1024):
        

        
        self.end2end=end2end
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(gpt2path, local_files_only=True)
            config = AutoConfig.from_pretrained(gpt2path, local_files_only=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model/tokenizer from {gpt2path}. "
                f"Make sure you download it first.\n{e}"
        )
            
            
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        

        # Figure out max context length from config (model-agnostic)
        if hasattr(config, "n_positions"):
            max_positions = config.n_positions
        elif hasattr(config, "max_position_embeddings"):
            max_positions = config.max_position_embeddings
        else:
            raise ValueError(f"Cannot determine max context length from {gpt2path}")

        # Enforce: prefix tokens + text tokens <= model limit
        self.max_seq_len = max_positions - prefix_length
        
        
        self.dataset=dataset
        if end2end:
            self.tokenize_dataset(token_batchsize)
            
        

    
    def tokenize_dataset(self, batch_size=1024):
        
        all_len=[]
        captions=self.dataset["summary"]
        tokenized = []
        for cap in tqdm(captions, desc="Tokenizing captions"):
            encoded = self.tokenizer.encode(cap, truncation=True, max_length=self.max_seq_len)
            tokenized.append(encoded)


        self.dataset = self.dataset.add_column("summary_tokens", tokenized)
        self.dataset = self.dataset.remove_columns(["summary"])
        

        
    def __len__(self) -> int:
        return len(self.dataset)

    def pad_tokens(self, item: int):
        tokens = self.dataset['summary_tokens'][item]
        tokens=torch.tensor(tokens, dtype=torch.int64)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.dataset['summary_tokens'][item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.dataset['summary_tokens'][item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int):
        tokens, mask = self.pad_tokens(item)
          
        if self.end2end:
            disassembly_info=self.dataset['disassembly'][item]
        else:
            prefix = self.dataset['disassembly_prefixes'][item]
            if self.normalize_prefix:
                prefix = prefix.float()
                prefix = prefix / prefix.norm(2, -1)
            disassembly_info=prefix
                                                     
        return tokens, mask, disassembly_info
    





class CustomCollator:
    def __init__(self, args):
        
        self.tokenizer = tokenizer.load(args.tokenizer)
        self.max_length=args.tokenizer_size

        self.tok_fast= PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)



    def __call__(self, batch):
        tokens, masks, disassembly_infos = zip(*batch)
        
        
        disassembly_batch=self.tok_fast(
            disassembly_infos,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
            )

        return {
            "tokens": torch.stack(tokens),
            "mask": torch.stack(masks),
            "disassembly_tokens": disassembly_batch['input_ids'],
            "disassembly_mask": disassembly_batch['attention_mask']
        }
    

