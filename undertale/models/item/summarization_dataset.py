from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
import multiprocessing
import numpy as np
from datasets import load_dataset




class SummarizerDataset(Dataset):
    
    
    def __init__(self, dataset, data_path, prefix_length, gpt2_type= "gpt2",
                 normalize_prefix=False,end2end=True,token_batchsize=1024):
        
        self.all_lengths = []
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        
        if end2end:
            self.tokenize_dataset(dataset,token_batchsize)

        else:
            
            if os.path.isfile(data_path):
                selfdataset = load_dataset("parquet", data_files=data_path)
            else:
                raise FileNotFoundError(f"File not found: {data_path}")
        
    def tokenize_batch(self, batch):
        
        input_ids = [self.tokenizer.encode(text) for text in batch['summary']]
        self.token_lengths.extend([len(ids) for ids in input_ids])
        
        return {"summary_tokens": input_ids}
    
    def tokenize_dataset(self, dataset, batch_size=1024):

        num_cpus = multiprocessing.cpu_count()

        print(f"Tokenizing with {num_proc} processes...")
        self.dataset = dataset.map(
            self.tokenize_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=["summary"]
        )

        all_len = np.array(self.token_lengths)
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10),int(all_len.max()))
        
    def __len__(self) -> int:
        return len(self.dataset)

    def pad_tokens(self, item: int):
        tokens = self.dataset['summary_tokens'][item]
        
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

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
          
        if end2end:
            disassembly_info=self.dataset['disassembly'][item]
        else:
            prefix = self.dataset['disassembly_prefixes'][item]
            if self.normalize_prefix:
                prefix = prefix.float()
                prefix = prefix / prefix.norm(2, -1)
            disassembly_info=prefix
                                                     
        
        return tokens, mask, disassembly_info
    


   
