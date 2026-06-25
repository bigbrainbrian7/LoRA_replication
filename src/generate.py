from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import build_test_dataset, tokenizer

test_dataset = build_test_dataset('data/src1_test.txt', 'data/test',)

model = GPT2LMHeadModel.from_pretrained('outputs/finetune/final')

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



dataloader = DataLoader(
    test_dataset,
    shuffle=False,
)

with open('outputs/generations/finetune.txt', 'w') as f:
    for input in dataloader:
        output = model.generate(**input)
        output = tokenizer.decode(output, skip_special_tokens=False)
        # print(output)
        output = output[0].split(tokenizer.eos_token)[1]
        f.write(output + '\n')

# generate_references()