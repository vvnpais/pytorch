import torch
import torch.nn as nn
import torch.nn.functional as F

#option 1 create nn modules

class NeuralNet(nn.module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        # nn.Sigmoid
        # nn.Softmax
        # nn.TanH
        # nn.LeakyReLU
        self.linear2=nn.Linear(hidden_size,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        out=self.linear1(x)
        out=self.relu(out)
        out=self.linear2(out)
        out=self.sigmoid(out)
        return out
    
# option2 use activation func in forward pass
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.linear2=nn.Linear(hidden_size,1)

    def forward(self,x):
        out=torch.relu(self.linear1(x))
        # F.relu
        # F.leaky_relu
        out=torch.sigmoid(self.linear2(out))
        # torch.softmax
        # torch.tanh  
        return out