import math
import os
import tiktoken
import torch
import sys
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F

from transformers import GPT2LMHeadModel


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query and value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Adding a mask (called a bias in the OpenAI/HF naming though)
        # .register_buffer: registers the mask as a buffer, so its not treated as a trainable parameter
        # torch.ones: create tensor of ones
        # torch.trill: apply a lower triangle operation
        # [[1 1 1],   [[1 0 0]
        #  [1 1 1], -> [1 1 0]
        #  [1 1 1]]    [1 1 1]]
        # torch.view: convert the same data to a different shape
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        """
        Method of multi-head self attention in a single pass
        """
        # x is a tensor of shape:
        # - Batch size
        # - Tokens (sequence length)
        # - embedding dimension (not sure why its called C))
        B, T, C = x.size()
        qkv = self.c_attn(
            x
        )  # Create query, key, value vectors via projecting linear layer
        q, k, v = qkv.split(
            self.n_embd, dim=2
        )  # torch.split: split tensor by which dimension

        # Efficient pass
        # - Making number of heads into a batch dimension so pytorhc treats B and n_heads as batches.
        # - Applies all operations on them in parallel.

        # Shaping q, v, t to (B, T, nh, C // nh)
        # transpose(1, 2) swaps the T and
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # # Operations
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # # Autoreggrssive mask, assures causal self attention
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # # Make sure it sums to one
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        
        # flash attention 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # Expand dimension
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # reduce dimension

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # The forward method controls the flow through the neural network
        x = x + self.attn(self.ln_1(x))  # Talking to eachother
        x = x + self.mlp(self.ln_2(x))  # Thinking by themselves
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

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
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme, use in both the beginning and end of the model.
        # We expect similar tokens to be close to eachother in the embedding space,
        # therefore the probability at the end of the transformer should be similar. 
        # Also we are neing 30% more efficient! (saving 30% of parameterr)
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # initialise linear layers 
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5   
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # set bias to zero
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_idx, targets=None):
        # token_idx is of shape (B, T)
        # B for Batch dimension. Batch dimension represents the number of samples (or sequences) processed simultaneously.
        # T for time dimension. Represents the sequence length or the number of time steps in each sample.
        B, T = token_idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        # torch.arange is a function in PyTorch that generates a 1-dimensional tensor (array) with values ranging from a start value to an end value
        pos = torch.arange(0, T, dtype=torch.long, device=token_idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(
            token_idx
        )  # token embeddings of shape(B, T, n_embd)
        x = tok_emb + pos_emb

        # forward blocks of the transformer
        for block in self.transformer.h:
            # The nn.Module class defines a __call__ method, which is automatically invoked when you treat an instance of the class like a function (e.g., block(x)).
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        # calculate the logits of what comes next (B, T + 1) out of vocab_size amount of tokens
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None: 
            # flattening the tensor
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Loads the model weights from huggingface
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        print("Loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        # GPT standard params
        config_args["vocab_size"] = 50304 # prettier number that 50257 (more powers of 2 fit in it)
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        # statedict for both models
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Not interested in the autoregressive mask
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # boring bit of code to transpose
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters that require grad.
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,}  parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,}  parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = False

        # python 3.12 hack, kernel fussion for AdamW
        # try:
        #     torch.optim.AdamW([], fused=True)  # Test with an empty parameter list
        #     fused_available = True
        # except TypeError:
        #     fused_available = False
        fused_available = True
        used_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {used_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=used_fused)
        return optimizer


# -------------------------------------------------------------------------------

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # loads tokens from disk and store them in memory
        with open("../input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs 
        y = (buf[1:]).view(B, T)  # targets
        # advance the posiiton by B * T 
        self.current_position += B * T

        # check if out of bounds, if so, reset it
        # +1 is used as it means we always have a label token
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
        

# --------------------------------------------------------------------------------
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# setup DDP
ddp = int(os.environ.get("RANK", -1)) != -1 # check if ddp run
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size= int(os.environ["WORLD_SIZE"])
    # make sure to use the approiate gpu
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    # make sure master_process is at 0
    master_process = ddp_rank ==0 # wil do logging and checkpointing
else:
    # else return to single GPU training
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# attempt to autodetect device
device_type = "cuda" if device.startswith("cuda") else "cpu"

total_batch_size = 524288 # 2**19, roughly 0.5 M of tokens
B = 4 # micro batch size
T = 1024 # sequence 

assert total_batch_size % (B * T * ddp_world_size) == 0 # make sure you can divide the total_batch_size by B*T*ddp_world_size
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) 
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumation steps: {grad_accum_steps}")

train_loader = DataLoader(B=8, T=1024)

# high = tf_32, setting all matrix multipication to utlize tf32 precision
torch.set_float32_matmul_precision("high")

# get logits
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 3e-4
min_lr = max_lr * 0.1
warm_up_steps = 10
max_steps = 50

# cosine learning rate 
def get_lr(it):
    # initialy start with linear warmup
    if it < warm_up_steps:
        return max_lr * (it+1) / warm_up_steps
    # if it > decay iters, return min_lr
    if it > max_steps:
        return min_lr
    # in between, use cosine decay down to min learning rate
    decay_ratio = (it - warm_up_steps) / (max_steps - warm_up_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (max_lr - min_lr)

    

# optimizing
# lr=3e-4 is good for debugging
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    
    # always start with a zero gradient
    optimizer.zero_grad()
    # Create batch
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # cast some operations to BF16, otherwise in FP32, as some operations are effected more.
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            # hacky solution here
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward() # accumulates gradient
    if ddp:
        # loss_accum averaged on all ranks
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # stop model shock, limit gradient to 1.0.
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # wait for all the work to finish.
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000

    # token count
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30

# prefix tokens
# tokenize input and convert to pytorch tensor
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
# add batch dimension as we expect to have a batch dimension then repeat it
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

# We have 5 batches of sequences that have a lenght of (B=5, T=8)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        # get logits
        logits = model(x)  # (B, T, vocab_size)
        # get the last logits
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get probabilities
        probs = F.softmax(logits, dim=-1) 
        # do top_k sampling
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities (sampling from top_k)
        ix = torch.multinomial(topk_probs, 1)
        # gather the corresponding indices (which tokens correspond to those probabilities)
        xcol = torch.gather(topk_indices, -1, ix)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
