import math
import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel

class LoraConv1D(nn.Module):

    def __init__(self, nf, nx, alpha, r):
        super().__init__()
        self.nf = nf
        self.nx = nx
        
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self.scaling = alpha / r

        self.a_weight = nn.Parameter(self.weight.new_empty(nx, r))
        self.b_q_weight = nn.Parameter(self.weight.new_zeros(r, nf//3))
        self.b_v_weight = nn.Parameter(self.weight.new_zeros(r, nf//3))
        nn.init.kaiming_uniform_(self.a_weight, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=0.1)

    def __repr__(self) -> str:
        return "LoraConv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x: torch.Tensor):
        size_out = x.size()[:-1] + (self.nf,)

        x = x.view(-1, x.size(-1))

        lora_hidden = self.lora_dropout(x) @ self.a_weight

        x = torch.addmm(self.bias, x, self.weight)

        q, k, v = x.split(self.nf//3, dim=-1)

        lora_q = (lora_hidden @ self.b_q_weight) * self.scaling
        lora_v = (lora_hidden @ self.b_v_weight) * self.scaling

        q = q + lora_q
        v = v + lora_v

        x = torch.cat((q,k,v), dim=-1)

        x = x.view(size_out)
        return x
    
    @classmethod
    def from_conv1d(cls, conv1d, alpha: int, r: int) -> 'LoraConv1D':
            """Create a LoraConv1D from an existing Conv1D, copying its weights."""
            # weights should be loaded automatically, not necessary
            nf = conv1d.nf
            nx = conv1d.weight.shape[0]
            lora = cls(nf=nf, nx=nx, alpha=alpha, r=r)
            # copy pretrained weights into the frozen weight
            lora.weight = nn.Parameter(conv1d.weight.detach().clone())
            lora.bias = nn.Parameter(conv1d.bias.detach().clone())
            lora.weight.requires_grad = False
            lora.bias.requires_grad = False
            return lora


def apply_lora(model: GPT2LMHeadModel, r: int = 4, alpha: int = 32) -> GPT2LMHeadModel:
    for block in model.transformer.h:
        original = block.attn.c_attn
        block.attn.c_attn = LoraConv1D.from_conv1d(original, alpha=alpha, r=r)
    return model


def freeze_non_lora(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    for name, param in model.named_parameters():
        if 'a_weight' not in name and 'b_q_weight' not in name and 'b_v_weight' not in name:
            param.requires_grad = False
    return model


def count_trainable_params(model: GPT2LMHeadModel):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Ratio:            {100 * trainable / total:.4f}%")