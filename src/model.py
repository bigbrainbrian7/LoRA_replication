import numpy as np
import json
import torch
import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from lora import apply_lora, freeze_non_lora, count_trainable_params


# def build_model(src='gpt2-medium', use_lora: bool = False) -> GPT2LMHeadModel:
#     model = GPT2LMHeadModel.from_pretrained(src)
    
#     if use_lora:
#         model = apply_lora(model, r=4, alpha=32)
#         model = freeze_non_lora(model)
    
#     count_trainable_params(model)
#     return model

def build_model(src='gpt2-medium', use_lora: bool = False) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    
    if use_lora:
        model = apply_lora(model, r=4, alpha=32)
        model = freeze_non_lora(model)
    
    if src != 'gpt2-medium':
        from safetensors.torch import load_file
        state_dict = load_file(f'{src}/model.safetensors')
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()
    
    # count_trainable_params(model)
    return model