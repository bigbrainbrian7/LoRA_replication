from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

import numpy as np
import json

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
#will eventually add build_model methods
