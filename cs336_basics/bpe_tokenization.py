import regex as re
from collections import Counter, defaultdict
import heapq
from functools import lru_cache
import multiprocessing
from dataclasses import dataclass
import pickle
import os

try:
    from .pretokenization_example import find_chunk_boundaries
except ImportError:
    from pretokenization_example import find_chunk_boundaries

PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@dataclass
class BPEConfig:
    input_path: str
    vocab_size: int
    special_tokens: list[str]
    use_multiprocessing: bool = True
    save_output: bool = False
    output_name: str = None

# Predefined configs
TINY_STORIES_SAMPLE_CONFIG = BPEConfig(
    input_path="/Users/ishanpatil1994/stanford-cs336-repos/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",
    vocab_size=1000,
    special_tokens=["<|endoftext|>"],
    save_output=True,
    output_name="tinystories_sample"
)

TINY_STORIES_TRAIN_CONFIG = BPEConfig(
    input_path="/Users/ishanpatil1994/stanford-cs336-repos/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    save_output=True,
    output_name="tinystories_train"
)

TINY_STORIES_TEST_CONFIG = BPEConfig(
    input_path="/Users/ishanpatil1994/stanford-cs336-repos/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    save_output=True,
    output_name="tinystories_test"
)

OPENWEBTEXT_VALID_CONFIG = BPEConfig(
    input_path="/Users/ishanpatil1994/stanford-cs336-repos/assignment1-basics/data/owt_valid.txt",
    vocab_size=32000,
    special_tokens=["<|endoftext|>"],
    save_output=True,
    output_name="openwebtext_valid"
)

OPENWEBTEXT_TRAIN_CONFIG = BPEConfig(
    input_path="/Users/ishanpatil1994/stanford-cs336-repos/assignment1-basics/data/owt_train.txt",
    vocab_size=32000,
    special_tokens=["<|endoftext|>"],
    save_output=True,
    output_name="openwebtext_train"
)

def save_bpe_results(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_name: str):
    """
    Save vocabulary and merges to pickle files in the output directory.
    
    Args:
        vocab: The tokenizer vocabulary mapping
        merges: List of BPE merges
        output_name: Base name for the output files (without extension)
    """
    output_dir = "/Users/ishanpatil1994/stanford-cs336-repos/assignment1-basics/cs336_basics/output"
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_path = os.path.join(output_dir, f"{output_name}_vocab.pkl")
    merges_path = os.path.join(output_dir, f"{output_name}_merges.pkl")
    
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    
    print(f"Saved vocabulary to {vocab_path}")
    print(f"Saved merges to {merges_path}")

def pretokenize_chunk(chunk: str, special_tokens: list[str]):
    # Split on all special tokens, so that no merging can occur across the text that they delimit
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    if escaped_special_tokens:
        regex_patten = f"({'|'.join(escaped_special_tokens)})"
        chunk_parts = re.split(regex_patten, chunk)
    else:
        chunk_parts = [chunk]
    result = defaultdict(int)
    for part in chunk_parts:
        if part not in special_tokens:
            for pre_token_match in re.finditer(PRETOKENIZATION_PATTERN, part):
                tuple_of_bytes = tuple(x.encode("utf-8") for x in pre_token_match.group())
                result[tuple_of_bytes] += 1
    return result
        

def process_chunk_worker(input_path: str, start: int, end: int, special_tokens: list[str]):
    """Worker function to process a single chunk of the file in parallel."""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize_chunk(chunk, special_tokens)


def run_pretokenization_parallel(input_path: str, special_tokens: list[str], use_multiprocessing: bool = True):
    """
    Run pretokenization on the input file, with optional multiprocessing.
    
    Args:
        input_path: Path to input text file
        special_tokens: List of special tokens to handle
        use_multiprocessing: Whether to use multiprocessing (default: True)
    
    Returns:
        Dictionary mapping token tuples to their frequencies
    """
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    if use_multiprocessing:
        # Create arguments for each worker process
        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk_args.append((input_path, start, end, special_tokens))
        
        # Process chunks in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            chunk_results = pool.starmap(process_chunk_worker, chunk_args)
    else:
        # Process chunks sequentially
        chunk_results = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk_result = process_chunk_worker(input_path, start, end, special_tokens)
            chunk_results.append(chunk_result)
    
    # Combine results from all processes/chunks
    pretokens_freq_table = defaultdict(int)
    for chunk_result in chunk_results:
        for pretoken in chunk_result:
            pretokens_freq_table[pretoken] += chunk_result[pretoken]
    
    return pretokens_freq_table

def merge_pretoken(pretoken: tuple[bytes, ...], merge_pair: tuple[bytes, bytes]):
    i = 0
    updated_pretoken = []
    while i < len(pretoken):
        if i == len(pretoken) - 1:
            updated_pretoken.append(pretoken[i])
            i += 1
        else:
            current_pair = (pretoken[i], pretoken[i+1])
            if current_pair == merge_pair:
                updated_pretoken.append(merge_pair[0] + merge_pair[1])
                i += 2
            else:
                updated_pretoken.append(pretoken[i])
                i += 1
    updated_pretoken = tuple(updated_pretoken)
    return updated_pretoken

class BPETokenizer:
    def __init__(self, pretoken_table: dict[tuple[bytes, ...], int]):
        self._pair_freq = defaultdict(int)
        self._pair_2_pretokens = defaultdict(set)
        self._pretoken_table = pretoken_table
        self._initialize_pair_counts()
        self._num_merges = 0
    
    def _initialize_pair_counts(self):
        for pretoken in self._pretoken_table:
            self._add_pairs(pretoken)

    def _remove_pairs(self, pretoken: tuple[bytes, ...]):
        if len(pretoken) < 2:
            return
        for idx in range(len(pretoken) - 1):
            pair = (pretoken[idx], pretoken[idx+1])
            self._pair_freq[pair] -= self._pretoken_table[pretoken]
            if self._pair_freq[pair] <= 0:
                del self._pair_freq[pair]
            if pretoken in self._pair_2_pretokens[pair]:
                self._pair_2_pretokens[pair].remove(pretoken)
    
    def _add_pairs(self, pretoken: tuple[bytes, ...]):
        if len(pretoken) < 2:
            return
        for idx in range(len(pretoken) - 1):
            pair = (pretoken[idx], pretoken[idx+1])
            self._pair_freq[pair] += self._pretoken_table[pretoken]
            self._pair_2_pretokens[pair].add(pretoken)

    def find_merges(self):
        if not self._pair_freq:
            return None
        max_freq = max(self._pair_freq.values())
        # print(self._num_merges, sorted([(x, y) for x, y in self._pair_freq.items()], key=lambda z : (-z[1], z[0]))[:10])
        max_freq_pairs = [pair for pair, freq in self._pair_freq.items() if freq == max_freq]
        merge_pair = max(max_freq_pairs)  # Deterministic: lexicographically largest
        if not merge_pair:
            return None
        pretokens_to_update = list(self._pair_2_pretokens[merge_pair])
        for pretoken in pretokens_to_update:
            self._remove_pairs(pretoken)
            updated_pretoken = merge_pretoken(pretoken, merge_pair)
            if updated_pretoken in self._pretoken_table:
                self._pretoken_table[updated_pretoken] += self._pretoken_table[pretoken]
            else:
                self._pretoken_table[updated_pretoken] = self._pretoken_table[pretoken]
            del self._pretoken_table[pretoken]
            self._add_pairs(tuple(updated_pretoken))
        self._num_merges += 1
        return (merge_pair[0], merge_pair[1])

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], use_multiprocessing: bool = True, save_output: bool = False, output_name: str = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given input text file.
    
    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary size 
                   (including the initial byte vocabulary, vocabulary items produced 
                   from merging, and any special tokens).
        special_tokens: A list of strings to add to the vocabulary. These special 
                       tokens do not otherwise affect BPE training.
        use_multiprocessing: Whether to use multiprocessing for pretokenization (default: True)
        save_output: Whether to save vocabulary and merges to pickle files (default: False)
        output_name: Base name for output files if saving (default: None)
    
    Returns:
        A tuple containing:
        - vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) 
                to bytes (token bytes).
        - merges: A list of BPE merges produced from training. Each list item is a tuple 
                 of bytes (<token1>, <token2>), representing that <token1> was merged with 
                 <token2>. The merges should be ordered by order of creation.
    """
    current_token_table = run_pretokenization_parallel(input_path, special_tokens, use_multiprocessing)
    print("Pre-tokenization has finished..")
    vocab = {}
    num_entries = 0
    for idx in range(256):
        vocab[num_entries] = bytes([idx])
        num_entries += 1
    for special_token in special_tokens:
        vocab[num_entries] = bytes(special_token, encoding="utf-8")
        num_entries += 1
    merges = []
    bpe_tokenizer = BPETokenizer(current_token_table)
    while len(vocab) < vocab_size:
        merge_pair = bpe_tokenizer.find_merges()
        if merge_pair is None:
            break
        vocab[num_entries] = merge_pair[0] + merge_pair[1]
        merges.append(merge_pair)
        num_entries += 1
    
    if save_output and output_name:
        save_bpe_results(vocab, merges, output_name)
    
    return vocab, merges

def train_bpe_from_config(config: BPEConfig) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer using a BPEConfig object.
    
    Args:
        config: BPEConfig object containing training parameters
    
    Returns:
        A tuple containing:
        - vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) 
                to bytes (token bytes).
        - merges: A list of BPE merges produced from training. Each list item is a tuple 
                 of bytes (<token1>, <token2>), representing that <token1> was merged with 
                 <token2>. The merges should be ordered by order of creation.
    """
    return train_bpe(config.input_path, config.vocab_size, config.special_tokens, config.use_multiprocessing, config.save_output, config.output_name)

if __name__ == '__main__':
    # Example usage with the sample config
    vocab, merges = train_bpe_from_config(OPENWEBTEXT_TRAIN_CONFIG)
