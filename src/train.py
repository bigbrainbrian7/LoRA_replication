import torch

from transformers import TrainingArguments, Trainer
from dataset import tokenizer, build_dataset
from model import model
from torch.nn.utils.rnn import pad_sequence

train_dataset = build_dataset('data/src1_train.txt', 'data/train',)
valid_dataset = build_dataset('data/src1_valid.txt', 'data/valid',)

def data2text_collator(features):
    input_ids = pad_sequence(
        [torch.tensor(f["input_ids"]) for f in features],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    labels = pad_sequence(
        [torch.tensor(f["labels"]) for f in features],
        batch_first=True,
        padding_value=-100
    )
    return {"input_ids": input_ids, "labels": labels}

training_args = TrainingArguments(
    output_dir="outputs/finetune/checkpoints",
    num_train_epochs=5,
    #gpu cannot handle batch size of 10, accumulate to simulate to replicate study hyperparameters
    #will unfortunately differ marginally
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=5,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data2text_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model('outputs/finetune/final')