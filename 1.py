import torch
import numpy as np

x=torch.tensor([4,5,6],requires_grad=True,dtype=torch.float32)
print(x)
y=x**4
print(y)
v=torch.tensor([1,1,1],dtype=torch.float32)
y.backward(v)
print(x.grad)