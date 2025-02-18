# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.11 Transformers
#     language: python
#     name: py311_transformers
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
# initialize weight matrices, grad set to false for example purposes
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

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
# calculate attention weights by sc
