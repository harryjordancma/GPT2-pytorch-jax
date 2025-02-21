# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.12
#     language: python
#     name: py312
# ---

# %% [markdown]
# # Chapter 3 - Coding attention mechanisms

# %% [markdown]
# ## 3.3 self attention w/o weights
#
# ### 3.3.1 For a single input vector

# %%
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
    )

# %%
# Compute attention scores
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

# %%
# calculate attention weights
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# %%
# softmax is better for normalising, naive implementation
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# %%
# pytorch implementation is better for dealing with numerical stability issues for big and small values
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# %%
# Calculating the second context vector
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)

# %% [markdown]
# ### 3.3.2 Computing attention weights for all input tokens

# %%
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

# %%
# for loops are slow, using matrix multipication
attn_scores = inputs @ inputs.T
print(attn_scores)

# %%
# normalise each row so that the values in each row sum to 1
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# %%
# verify row_2 sums up to 1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# %%
# calculate context vectors for all rows
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# %%
# comparing both context calculated by both individual and all
print("Previous 2nd context vector:", context_vec_2)

# %% [markdown]
# ## 3.4 Implementing self-attention with trainable weights

# %%
# define variables
x_2 = inputs[1] # second input vector
d_in = inputs.shape[1] # input dimensions
d_out = 2 # output dimensions

# %%
inputs

# %%
# initialize weight matrices, grad set to false for example purposes
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# %%
W_key

# %%
# computer query, key and value vectors
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# %%
# computing all keys and values
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape", values.shape)

# %%
# Compute attention score for 2nd input vector
keys_2 = keys[1]
attn_scores_22 = query_2.dot(keys_2)
print(attn_scores_22)

# %%
# calculate attention scores via matrix multiplication
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# %%
# calculate attention weights by dividing the them by the sqrt of the embedding dimensions
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# %%
# calculate context vector for 2nd input vector
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# %% [markdown]
# ## 3.4.2 Implementing a compact self-attention python class

# %%
# First version of the self attention class
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


# %%
# Using class
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


# %%
# Using nn.Linear instead of random nn.Parameter as has a more optimized weight initializiation scheme
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
        


# %%
# Using SelfAttention_v2
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# %% [markdown]
# #### Exercise 3.1 - Transfering weights

# %%
# Assigning weights
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)

# %%
# Confirming output is the same
sa_v1(inputs)

# %% [markdown]
# ## 3.5 Hiding words with causal attention

# %%
# Calculate attention weights as before
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# %%
# Creating mask wher values above diagonal are zero
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# %%
# multiply mask w/ attention weights
masked_simple = attn_weights * mask_simple
print(masked_simple)

# %% [markdown]
# Applying the softmax before will cause info leakage, however if the result is renormalised, the original softmax wont effect anything. 

# %%
# renormalise the attention weights to sum up to 1 by dividing by the sum
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# %% [markdown]
# #### More efficient masking using softmax property
# We wont need to renormalise if we apply the mask to the scores then use the softmax to mask.

# %%
# Make this more efficient using softmax property
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# %%
# applying the softmax now will cause the inf to go to 0
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# %% [markdown]
# ### 3.5.2 Masking additional attention weights w/ dropout

# %%
# Using dropout to prevent overfitting
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

# %% [markdown]
# Values are scaled up from 1 to 2 (1/0.5 = 2). This is done to maintain balance during both training and inference

# %%
# Applying dropout to weight matrix
torch.manual_seed(123)
print(dropout(attn_weights))

# %% [markdown]
# ### 3.5.3 Implementing a compact causal attention class
#

# %%
# simulate batch of inputs
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)


# %%
# Creating causal attention class
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # buffers are registered so that they are moved to to correct device when needed
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # batch dimension unaffected,
        # in place operation
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
        


# %%
# using CausalAttention class
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)


# %% [markdown]
# ## 3.6 Extending single head attention to multi-head attention

# %%
# wrapper class for multi head attention
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                    dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
            
        

# %%
# Using multihead attention wrapper class
torch.manual_seed(123)
context_length = batch.shape[1] # number of tokens
print(f"Number of tokens")
d_in, d_out = 3, 2

# %%
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# %% [markdown]
# #### Example 3.2 Returning two-dimensional emebdding vectors

# %%
context_length

# %%
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# %%
context_length


# %% [markdown]
# ## 3.6.2 Implementing multi-head attention with weight splits

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split matrix with num_heads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # create mask
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax
        
        

        
