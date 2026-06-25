import os
import copy
import torch

from typing import cast
from transformers import GPT2Tokenizer
from datasets import Dataset, load_from_disk

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

def build_dataset(
        file_path: str,
        cache_dir: str,
        block_size: int = 512,
) -> Dataset:
    if os.path.exists(cache_dir):
        print(f"Loading cache dataset from {cache_dir}")
        return cast(Dataset,load_from_disk(cache_dir))
    
    assert os.path.isfile(file_path), f"Input data file path {file_path} not found"
    with open(file_path, encoding='utf-8') as f:
        lines = [ 
            line.split('||') for line in f.read().splitlines() 
            if len(line) > 0 
            and not line.isspace() 
            and len(line.split('||')) == 2 
        ]

        src_lines, tgt_lines = zip(*lines)

        edited_sents = []

        for src, tgt in zip(src_lines, tgt_lines):
            sent = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)
            edited_sents.append(sent)

        batch_encoding = tokenizer(
            edited_sents,
            add_special_tokens=False,
            truncation=True,
            max_length=block_size
        )

        input_ids = batch_encoding['input_ids']

        labels = copy.deepcopy(input_ids)

        separator_id = tokenizer(tokenizer.bos_token, add_special_tokens=False)['input_ids'][0]

        for i, elem in enumerate(input_ids):
            sep_idx = elem.index(separator_id) + 1
            # special symbol fro cross entropy loss. need to confirm
            labels[i][:sep_idx] = [-100] * sep_idx

        print(edited_sents[0])
        print(input_ids[0])
        print(labels[0])


        dataset_dict = {
            'input_ids': input_ids,
            'labels': labels
        }

        hf_dataset = Dataset.from_dict(dataset_dict)
        hf_dataset.set_format('torch')

        hf_dataset.save_to_disk(cache_dir)
        print(f"Dataset cached to {cache_dir}")
        return hf_dataset
    

# Redundant content
# How to properly remove duplicate inputs to feed into llm but also preserve different paraphrasing for e2e metrics?
def build_test_dataset(
        file_path: str,
        cache_dir: str,
        block_size: int = 512,
) -> Dataset:
    if os.path.exists(cache_dir):
        print(f"Loading cache dataset from {cache_dir}")
        return cast(Dataset,load_from_disk(cache_dir))
    
    assert os.path.isfile(file_path), f"Input data file path {file_path} not found"
    with open(file_path, encoding='utf-8') as f:
        lines = [ 
            line.split('||') for line in f.read().splitlines() 
            if len(line) > 0 
            and not line.isspace() 
            and len(line.split('||')) == 2 
        ]

        #in theory, could be written as 
        # src_lines + bos token -> tokenize -> inputs
        # however, i am worried about the tokenizing boundaries.
        # attention mask not passed in, creates warnings when dict is unpacked for generation

        src_lines, tgt_lines = zip(*lines)

        edited_sents = []

        for src, tgt in zip(src_lines, tgt_lines):
            sent = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)
            edited_sents.append(sent)

        batch_encoding = tokenizer(
            edited_sents,
            add_special_tokens=False,
            truncation=True,
            max_length=block_size
        )

        input_ids = batch_encoding['input_ids']

        # labels = copy.deepcopy(input_ids)

        separator_id = tokenizer(tokenizer.bos_token, add_special_tokens=False)['input_ids'][0]

        for i, elem in enumerate(input_ids):
            sep_idx = elem.index(separator_id) + 1
            # special symbol fro cross entropy loss. need to confirm
            input_ids[i] = input_ids[i][:sep_idx]


        seen = set()
        unique = []

        for id in input_ids:
            t = tuple(id)
            if t not in seen:
                seen.add(t)
                unique.append(id)

        input_ids = unique

        # print(edited_sents[0])
        # print(type(input_ids))
        # print(type(input_ids[0]))
        # print(input_ids[0])
        # print(input_ids[1])
        # print(labels[0])

        dataset_dict = {
            'input_ids': input_ids,
            # 'labels': labels
        }

        hf_dataset = Dataset.from_dict(dataset_dict)
        hf_dataset.set_format('torch')

        hf_dataset.save_to_disk(cache_dir)
        print(f"Dataset cached to {cache_dir}")
        return hf_dataset

# train_dataset = build_dataset('data/src1_train.txt', 'data/train',)
# valid_dataset = build_dataset('data/src1_valid.txt', 'data/valid',)
# test_dataset = build_test_dataset('data/src1_test.txt', 'data/test',)