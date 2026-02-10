import os
import multiprocessing
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter, defaultdict
import time
import json
from cs336_basics.multi_process import process_chunk, PAT
from typing import Iterable

def get_bytes_to_unicode():
    # 这个函数是为了创建一个从 byte (0-255) 到 unicode 字符的映射。
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class BPETokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        # input
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # caculate
        self.bytes_pairs_counter: Counter[tuple[bytes, bytes]] = Counter()
        self.bytes_paris_source_word: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

        # output
        self.vocab: dict[int, bytes] = {}
        # init vocab with single byte tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        if special_tokens:
            for s_token in special_tokens:
                if s_token.encode('utf-8') not in self.vocab.values():
                    self.vocab[len(self.vocab)] = s_token.encode('utf-8')

        self.merges: list[tuple[bytes, bytes]] = []

    def multiprocess_pretokenize(self, num_workers: int = None) -> Counter:
        num_processes = multiprocessing.cpu_count() if num_workers is None else num_workers
        tasks = []

        # --- 优化分块逻辑 (见下文) ---
        with open(self.input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((self.input_path, start, end, self.special_tokens))

        print(f"Starting pre-tokenization with {num_processes} workers on {len(tasks)} chunks...")

        # 最终的总计数器
        total_counter = Counter()

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(process_chunk, tasks)
            
            # 合并结果 (Reduce 阶段)
            print("Merging counters from workers...")
            for worker_counter in results:
                total_counter.update(worker_counter)

        print(f"Total unique tokens: {len(total_counter)}")
        return total_counter


    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """ Train BPE tokenizer """

        # 1. pretokenize the input text
        pre_token_list = self.multiprocess_pretokenize()
        for word, count in pre_token_list.items():
            if len(word) == 1:
                continue
            for i in range(len(word) - 1):
                byte_pair = (word[i], word[i + 1])
                self.bytes_pairs_counter[byte_pair] += count
                self.bytes_paris_source_word[byte_pair].add(word)
        
        # 2. max & lexicographically greater, then merge
        while len(self.vocab) < self.vocab_size:
            # print(f"Current Vocab Size: {len(self.vocab)}")
            self.bytes_pairs_counter = +self.bytes_pairs_counter 
            
            if not self.bytes_pairs_counter:
                print("No More Byte Pairs TO Merge...")
                break

            max_pair = max(self.bytes_pairs_counter, key=lambda k: (self.bytes_pairs_counter[k], k))
            
            source_words = self.bytes_paris_source_word[max_pair]
            
            # 使用 list 快照遍历
            for word in list(source_words):
                word_freq = pre_token_list[tuple(bytes([b]) for b in b"".join(word))]
                
                new_word = []
                i = 0
                while i < len(word):
                    # 检查是否匹配 max_pair
                    if i < len(word) - 1 and word[i] == max_pair[0] and word[i + 1] == max_pair[1]:
                        
                        if i > 0:
                            prev_char = new_word[-1]
                            self.bytes_pairs_counter[(prev_char, word[i])] -= word_freq
                            self.bytes_pairs_counter[(prev_char, b''.join(max_pair))] += word_freq
                        
                        if i < len(word) - 2:
                            next_char = word[i + 2]
                            self.bytes_pairs_counter[(word[i+1], next_char)] -= word_freq
                            self.bytes_pairs_counter[(b''.join(max_pair), next_char)] += word_freq

                        new_word.append(b''.join(max_pair))
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                
                new_word = tuple(new_word)

                for j in range(len(word) - 1):
                    self.bytes_paris_source_word[(word[j], word[j+1])].discard(word)
                    
                for k in range(len(new_word) - 1):
                    self.bytes_paris_source_word[(new_word[k], new_word[k+1])].add(new_word)

            self.bytes_pairs_counter.pop(max_pair, None)
            self.merges.append(max_pair)
            self.vocab[len(self.vocab)] = b''.join(max_pair)
        
        return self.vocab, self.merges
    
def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    #Path definition
    vocab_path = os.path.join(save_dir, "vocab.json")
    merges_path = os.path.join(save_dir, "merges.txt")
    byte_encoder = get_bytes_to_unicode()


    # ============================
    # 1. 保存 vocab.json
    # ============================
    # 目标格式: {"token": id, ...}
    # 你的格式: {id: b'token', ...}
    # 转换步骤: 反转 key-value，并将 bytes 解码为 string
    vocab_to_save = {}
    
    for token_id, token_bytes in vocab.items():
        try:
            # 通常使用 utf-8 解码。
            # 如果你的 BPE 处理了任意二进制数据，可能需要用 'latin-1' 防止报错
            # token_str = token_bytes.decode("utf-8", errors="replace")
            mapped_token = "".join(byte_encoder[b] for b in token_bytes)
            vocab_to_save[mapped_token] = token_id
        except Exception as e:
            print(f"Error decoding token {token_bytes}: {e}")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_to_save, f, ensure_ascii=True, indent=None)

    print(f"Saved vocab to {vocab_path}")

    # ============================
    # 2. 保存 merges.txt
    # ============================
    # 格式: 每行两个 token，用空格隔开
    with open(merges_path, "w", encoding="utf-8") as f:
        # 通常 HuggingFace 的 tokenizer 会在第一行加个版本注释，不加也可以
        # f.write("#version: 0.2\n")
        
        for p1, p2 in merges:
            # s1 = p1.decode("utf-8", errors="replace")
            # s2 = p2.decode("utf-8", errors="replace")
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")
            
    print(f"Saved merges to {merges_path}")


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens = sorted_tokens
        else:
            self.special_tokens = []
        self.token_to_idx = {token: idx for idx, token in vocab.items()}
        self.ranks = {merge: i for i, merge in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        byte_encoder = get_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
            vocab = {}
            for token_str, idx in raw_vocab.items():
                token_bytes = bytes(byte_decoder[char] for char in token_str)
                vocab[idx] = token_bytes
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\r\n') # 去掉换行符
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    print(f"Skipping invalid merge line: {line}")
                    continue
                p1_str, p2_str = parts
                p1_bytes = bytes(byte_decoder[char] for char in p1_str)
                p2_bytes = bytes(byte_decoder[char] for char in p2_str)
                merges.append((p1_bytes, p2_bytes))                

        return cls(vocab, merges, special_tokens)
    
    def pretokenize(self, text:str) -> list[bytes]:
        tokens = []
        if self.special_tokens:
            pattern = '|'.join(re.escape(token) for token in self.special_tokens)
            split_chunks = re.split(r"(" + pattern + r")", text)
        else:
            split_chunks = [text]
        
        for split_chunk in split_chunks:
            if split_chunk in self.special_tokens:
                tokens.append(split_chunk.encode('utf-8'))
            else:
                for match in PAT.finditer(split_chunk):
                    token = match.group(0).encode('utf-8')
                    tokens.append(token)
        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = self.pretokenize(text)
        token_ids = []
        for token in tokens:
            if len(token) == 1 or token.decode('utf-8') in self.special_tokens:
                token_ids.append(self.token_to_idx[token])
            else:
                sub_tokens = [bytes([b]) for b in token]
                while len(sub_tokens) > 1:
                    pairs = [(sub_tokens[i], sub_tokens[i + 1]) for i in range(len(sub_tokens) - 1)]
                    best_pair = None
                    best_rank = float('inf')
                    for pair in pairs:
                        if pair in self.ranks and self.ranks[pair] < best_rank:
                            best_rank = self.ranks[pair]
                            best_pair = pair
                    if best_pair is None:
                        break
                    first, second = best_pair
                    new_sub_tokens = []
                    i = 0
                    while i < len(sub_tokens):
                        if i < len(sub_tokens) - 1 and sub_tokens[i] == first and sub_tokens[i + 1] == second:
                            new_sub_tokens.append(b''.join(best_pair))
                            i += 2
                        else:
                            new_sub_tokens.append(sub_tokens[i])
                            i += 1
                    sub_tokens = new_sub_tokens
                for sub_token in sub_tokens:
                    token_ids.append(self.token_to_idx[sub_token])
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        byte_string = b''.join(tokens)
        return byte_string.decode('utf-8', errors='replace')

if __name__ == "__main__":

    # Patr1 训练bpe tokenizer
    # input_path = "data/test_bpe.txt"
    start_time = time.perf_counter()
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    bpe_tokenizer = BPETokenizer(input_path, vocab_size, special_tokens)
    vocab, merges = bpe_tokenizer.train_bpe()
    end_time = time.perf_counter()
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    print("Final Vocab Size:", len(vocab))
    save_tokenizer(vocab, merges, "output/owt_train")
    top_2 = sorted(vocab.items(), key=lambda x: (len(x[1]), x[1]), reverse=True)[:2]
    if len(top_2) >= 1:
        print("最大:", top_2[0])
    if len(top_2) >= 2:
        print("第二大:", top_2[1])

    # Part2  encoder decoder
    # vocab_filepath = "output/train/vocab.json"
    # merges_filepath = "output/train/merges.txt"
    # test_text = """u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
    #     <|endoftext|>
    #     Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
    #     Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
    #     Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
    #     They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
    #     <|endoftext|>"""
    # tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=["<|endoftext|>"])
    # res = tokenizer.encode(test_text)
    # decode_res = tokenizer.decode(res)
    # print("Encoded result:", res)
    # print("Decoded result:", decode_res)