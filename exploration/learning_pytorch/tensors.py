# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.11
#     language: python
#     name: py311
# ---

# %%
import torch
import numpy as np

# %% [markdown]
# ## Initialise a tensor

# %% [markdown]
# ### Directly from data

# %%
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# %%
## from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# %%
# From another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# %%
# With random or constant values
shape = (2,4,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# %% [markdown]
# ## Attributes of a Tensor

# %%
tensor = torch.rand(3,4)

print(f"Shape {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %% [markdown]
# ## Operations on Tensors

# %%
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# %%
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# %%
# Join tensors
t1 = torch.cat([tensor, 2 *tensor, 3*tensor], dim=1)
print(t1)

# %%
# Arithmetic
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# %%
# Compute element wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# %%
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# %%
# in place operations
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# %%
# bridge with NumPy
# changes to one reflect on the other as they have the same memory locations on a cpu

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# %%
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
