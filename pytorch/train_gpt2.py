from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query and value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Adding a mask (called a bias in the OpenAI/HF naming though
        # .register_buffer: registers the mask as a buffer, so its not treated as a trainable parameter
        # torch.ones: create tensor of ones
        # torch.trill: apply a lower triangle operation
        # [[1 1 1],   [[1 0 0]
        #  [1 1 1], -> [1 1 0]
        #  [1 1 1]]    [1 1 1]]
        # torch.view: convert the same data to a different shape
        self.register_buffer("bias", 
                             torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Method of multi-head self attention in a single pass
        """
        # x is a tensor of shape: 
        # - Batch size 
        # - Tokens (sequence length)
        # - embedding dimension (not sure why its called C))
        B, T, C = x.size()
        qkv = self.c_attn(x) # Create query, key, value vectors via projecting linear layer
        q, k, v = qkv.split(self.n_embd, dim=2) # torch.split: split tensor by which dimension

        """
        Efficient pass
        - Making number of heads into a batch dimension so pytorhc treats B and n_heads as batches.
        - Applies all operations on them in parallel. 
        """
        # Shaping q, v, t to (B, T, nh, C // nh)
        # transpose(1, 2) swaps the T and
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Operations
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Autoreggrssive mask, assures causal self attention
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        
        # Make sure it sums to one
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y
        
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # Expand dimension
        self.gelu = nn.GELU(approximate = "tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # reduce dimension

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # The forward method controls the flow through the neural network
        x = x + self.attn(self.ln_1(x)) # Talking to eachother
        x = x + self.mlp(self.ln_2(x)) # Thinking by themselves
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # nn.ModuleDict, allows us to access the submodules with keys 
        # wte (weights of token embeddings)
        # wpe (weights of the position embeddings)
        # h (hidden layers), a modulelist that we can access with int.
        # ln_f (layer normal final)
        # lm_head (language model head, projects to vocab size (50257))
        # no bias, therefore y = mx + 0
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        