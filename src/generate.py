from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import build_test_dataset, tokenizer
from model import build_model

test_dataset = build_test_dataset('data/src1_test.txt', 'data/test',)

def generate_references():
    with open('outputs/generations/references.txt', 'w') as output:
        with open('data/src1_test.txt') as f:
            prev = None
            for l in f.read().splitlines():
                mr, phrase = l.split('||')
                if mr != prev:
                    output.write('\n')
                    prev = mr
                output.write(phrase+'\n')

def write_outputs(dataloader, model: GPT2LMHeadModel, file):
    with open(file, 'w') as f:
        with open('outputs/generations/bruh.txt', 'w') as g:
            for input in dataloader:
                output = model.generate(**input, max_new_tokens=100)
                output = tokenizer.decode(output, skip_special_tokens=False)
                # print(output)
                g.write(output[0] + '\n')
                output = output[0].split(tokenizer.eos_token)[1]
                f.write(output + '\n')



dataloader = DataLoader(
    test_dataset,
    shuffle=False,
)


# generate_references()
# write_outputs(dataloader, build_model('outputs/finetune/final'), 'outputs/generations/finetune.txt')
write_outputs(dataloader, build_model('outputs/lora/final', True), 'outputs/generations/lora.txt')

