import torch
from torch.utils.data import Dataset, DataLoader
import html
import os
import pandas as pd
import re
from collections import Counter
import multiprocessing
from functools import partial

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        # Standard special tokens
        self.add_word('<unk>')
        self.add_word('<pad>')
        self.add_word('<s>')  # Start of Sentence (SOS)
        self.add_word('</s>') # End of Sentence (EOS)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def lookup_token(self, token):
        # 1. Try exact match
        if token in self.word2idx:
            return self.word2idx[token]
        
        # 2. Try lowercase match
        token_lower = token.lower()
        if token_lower in self.word2idx:
            return self.word2idx[token_lower]
            
        # 3. Fallback to <unk>
        return self.word2idx.get('<unk>')

    def lookup_index(self, index):
        return self.idx2word.get(index, '<unk>')

def load_vocab(file_path):
    vocab = Vocabulary()
 
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                vocab.add_word(word)
    return vocab

def find_html_entities(file_path):
    entities = set()
    # Regex for named entities (&name;) and numeric entities (&#123; or &#x12a;)
    entity_pattern = re.compile(r'&[a-zA-Z]+;|&#[0-9]+;|&#x[0-9a-fA-F]+;')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            matches = entity_pattern.findall(line)
            entities.update(matches)
    return entities

def process_line(line):
    tokens = line.strip().split()
    return tokens

class IWSLTDataset(Dataset):
    def __init__(self, en_path, vi_path, en_vocab, vi_vocab, max_length=100, use_gpu_processing=False):
        self.en_vocab = en_vocab
        self.vi_vocab = vi_vocab
        self.pairs = []
        self.original_pairs = [] # Store some for visualization
        
        print(f"Loading data from {en_path} and {vi_path}...")
        
        with open(en_path, 'r', encoding='utf-8') as f:
            lines_en = f.readlines()
        with open(vi_path, 'r', encoding='utf-8') as f:
            lines_vi = f.readlines()
            

        num_workers = multiprocessing.cpu_count() if use_gpu_processing else 1
        print(f"Processing with {num_workers} workers...")
        
        if num_workers > 1:
            with multiprocessing.Pool(num_workers) as pool:
                tokens_en_list = pool.map(process_line, lines_en)
                tokens_vi_list = pool.map(process_line, lines_vi)
        else:
            tokens_en_list = [process_line(line) for line in lines_en]
            tokens_vi_list = [process_line(line) for line in lines_vi]

        for i, (tokens_en, tokens_vi) in enumerate(zip(tokens_en_list, tokens_vi_list)):
            if len(tokens_en) > max_length or len(tokens_vi) > max_length or len(tokens_en) == 0 or len(tokens_vi) == 0:
                continue
            
            indices_en = [en_vocab.lookup_token(t) for t in tokens_en] + [en_vocab.lookup_token('</s>')]
            indices_vi = [vi_vocab.lookup_token('<s>')] + [vi_vocab.lookup_token(t) for t in tokens_vi] + [vi_vocab.lookup_token('</s>')]
            
            self.pairs.append((indices_en, indices_vi))
            
            if i < 10:
                self.original_pairs.append((lines_en[i].strip(), lines_vi[i].strip()))

        print(f"Loaded {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
    
    def get_dataframe(self):
        """
        Returns a pandas DataFrame describing the processed dataset.
        """
        data = []
        for src_idx, tgt_idx in self.pairs:
            data.append({
                'src_len': len(src_idx),
                'tgt_len': len(tgt_idx),
                # We can add more stats here
            })
        return pd.DataFrame(data)

    def analyze_vocab_distribution(self, indices_list, vocab):

        all_indices = [idx for seq in indices_list for idx in seq]
        counter = Counter(all_indices)
        
        word_counts = {vocab.lookup_index(idx): count for idx, count in counter.items()}
        return word_counts

    def show_processed_vs_original(self, num_samples=5):
        print(f"\n--- Processed vs Original (First {num_samples} samples) ---")
        for i in range(min(num_samples, len(self.original_pairs))):
            orig_en, orig_vi = self.original_pairs[i]
            proc_en_idx, proc_vi_idx = self.pairs[i]
            
            proc_en = " ".join([self.en_vocab.lookup_index(idx) for idx in proc_en_idx])
            proc_vi = " ".join([self.vi_vocab.lookup_index(idx) for idx in proc_vi_idx])
            
            print(f"Sample {i+1}:")
            print(f"  Orig En: {orig_en}")
            print(f"  Proc En: {proc_en}")
            print(f"  Orig Vi: {orig_vi}")
            print(f"  Proc Vi: {proc_vi}")
            print("-" * 20)

    def save(self, file_path):
        """
        Saves the processed dataset (pairs) to a file.
        """
        print(f"Saving dataset to {file_path}...")
        torch.save({
            'pairs': self.pairs,
            'en_vocab': self.en_vocab,
            'vi_vocab': self.vi_vocab
        }, file_path)
        print("Dataset saved.")

    @classmethod
    def load(cls, file_path):
        """
        Loads a processed dataset from a file.
        """
        print(f"Loading dataset from {file_path}...")
        # weights_only=False is required to load custom classes like Vocabulary
        # In a production environment with untrusted data, use add_safe_globals instead.
        checkpoint = torch.load(file_path, weights_only=False)
        
        # Create an empty instance
        # We need to bypass __init__ because it requires paths
        instance = cls.__new__(cls) 
        instance.pairs = checkpoint['pairs']
        instance.en_vocab = checkpoint['en_vocab']
        instance.vi_vocab = checkpoint['vi_vocab']
        instance.original_pairs = [] # Original pairs are not saved to save space
        
        print(f"Loaded {len(instance.pairs)} pairs.")
        return instance

    def export_to_text(self, en_path, vi_path):
        """
        Exports the processed dataset to text files (converting indices back to tokens).
        """
        print(f"Exporting processed dataset to {en_path} and {vi_path}...")
        with open(en_path, 'w', encoding='utf-8') as f_en, \
             open(vi_path, 'w', encoding='utf-8') as f_vi:
            
            for en_idx, vi_idx in self.pairs:
                # Convert indices to tokens
                en_tokens = [self.en_vocab.lookup_index(idx) for idx in en_idx]
                vi_tokens = [self.vi_vocab.lookup_index(idx) for idx in vi_idx]
                
                f_en.write(" ".join(en_tokens) + "\n")
                f_vi.write(" ".join(vi_tokens) + "\n")
        print("Export complete.")



def build_dataloaders(config):
    """
    Builds dataloaders for train, validation, and test sets.
    
    Args:
        config (dict): Configuration dictionary containing paths and parameters.
            Expected keys:
            - dataset_dir: Path to dataset directory
            - batch_size: Batch size
            - max_length: Max sequence length
            - num_workers: Number of workers for DataLoader
    
    Returns:
        train_loader, val_loader, test_loader, en_vocab, vi_vocab
    """
    dataset_dir = config['dataset_dir']
    batch_size = config.get('batch_size', 32)
    max_length = config.get('max_length', 100)
    num_workers = config.get('num_workers', 0)
    task = config.get('task', 'train')
    
    # Paths
    en_vocab_path = os.path.join(dataset_dir, "vocab.en.txt")
    vi_vocab_path = os.path.join(dataset_dir, "vocab.vi.txt")
    
    train_en_path = os.path.join(dataset_dir, "train.en.txt")
    train_vi_path = os.path.join(dataset_dir, "train.vi.txt")
    
    val_en_path = os.path.join(dataset_dir, "tst2012.en.txt")
    val_vi_path = os.path.join(dataset_dir, "tst2012.vi.txt")
    
    test_en_path = os.path.join(dataset_dir, "tst2013.en.txt")
    test_vi_path = os.path.join(dataset_dir, "tst2013.vi.txt")

    # processed_en_text_path = os.path.join(dataset_dir, "train.processed.en.txt")
    # processed_vi_text_path = os.path.join(dataset_dir, "train.processed.vi.txt")
    
    #
    print("Loading vocabularies...")
    en_vocab = load_vocab(en_vocab_path)
    vi_vocab = load_vocab(vi_vocab_path)
    

    if task == "train":
        print("Creating Train Dataset...")
        train_dataset = IWSLTDataset(train_en_path, train_vi_path, en_vocab, vi_vocab, max_length, use_gpu_processing=True)
        
        print("Creating Validation Dataset...")
        # Validation usually doesn't need filtering by length as strictly, but for batching it helps.
        val_dataset = IWSLTDataset(val_en_path, val_vi_path, en_vocab, vi_vocab, max_length, use_gpu_processing=True)
    else:
        print("Creating Test Dataset...")
        test_dataset = IWSLTDataset(test_en_path, test_vi_path, en_vocab, vi_vocab, max_length, use_gpu_processing=True)

    # train_dataset.export_to_text(
    #     processed_en_text_path,
    #     processed_vi_text_path
    # )

    # 3. Create DataLoaders
    pad_idx = en_vocab.lookup_token('<pad>') # Assuming same pad index for both or handled
    # Note: vi_vocab might have different pad index if we didn't force it. 
    # Our Vocabulary class adds <pad> at the beginning, so it should be consistent if added in same order.
    # But safer to get specific pad indices.
    
    # Actually collate_fn needs to know which pad_idx to use. 
    # Usually we use the same index if possible, or we pad src and tgt separately.
    # Our collate_fn takes one pad_idx. Let's check if they are same.
    en_pad = en_vocab.lookup_token('<pad>')
    vi_pad = vi_vocab.lookup_token('<pad>')
    
    if en_pad != vi_pad:
        print(f"Warning: <pad> index differs (En: {en_pad}, Vi: {vi_pad}). Using En pad for both in simple collate.")
        # We should update collate_fn to handle separate pad indices if needed.
    
    # Update collate_fn to handle separate pad indices
    collate_lambda = lambda b: collate_fn_separate(b, en_pad, vi_pad)

    train_loader = None
    val_loader = None
    test_loader = None
    if task == 'train':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_lambda)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_lambda)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_lambda)
    
    return train_loader, val_loader, test_loader, en_vocab, vi_vocab

def collate_fn_separate(batch, src_pad_idx, tgt_pad_idx):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(src_sample))
        tgt_batch.append(torch.tensor(tgt_sample))
        
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=src_pad_idx, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=tgt_pad_idx, batch_first=True)
    
    return src_batch, tgt_batch


