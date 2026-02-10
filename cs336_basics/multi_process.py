import os
import multiprocessing
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter, defaultdict
import time
import json

# gpt2
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# def process_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> list[tuple[bytes, ...]]:
#     """ Process a chunk of the input file for pretokenization """
#     tokens = []
#     with open(input_path, 'rb') as f:
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Split by special tokens
#         if special_tokens:
#             pattern = '|'.join(re.escape(token) for token in special_tokens)
#             split_chunks = re.split(pattern, chunk)
#         else:
#             split_chunks = [chunk]
#         for split_chunk in split_chunks:
#             for match in PAT.finditer(split_chunk):
#                 token = match.group(0).encode('utf-8')
#                 tokens.append(tuple(bytes([b]) for b in token))
#     return tokens

def process_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter:
    """ 
    优化后：直接返回 Counter，而不是 list。
    大幅减少进程间通信的数据量。
    """
    # 使用局部 Counter，比全局 Counter 快
    local_counter = Counter()
    
    with open(input_path, 'rb') as f:
        f.seek(start)
        # 读取并解码
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # 1. 处理 Special Tokens 切分
    if special_tokens:
        # 使用 regex 的 split 通常比 re 快
        pattern = '|'.join(re.escape(token) for token in special_tokens)
        parts = re.split(pattern, text)
    else:
        parts = [text]

    # 2. 正则分词并统计
    for part in parts:
        # finditer 返回的是迭代器，内存友好
        for match in PAT.finditer(part):
            token_bytes = match.group(0).encode('utf-8')
            
            # 转换成 tuple(bytes, ...) 格式
            # 这一步是计算密集型的，放在子进程做最好
            token_key = tuple(bytes([b]) for b in token_bytes)
            
            # 直接计数
            local_counter[token_key] += 1
            
    return local_counter