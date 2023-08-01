import torch
import torch.nn as nn
import numpy as np

# def softmax(x):
#     return np.exp(x)/np.sum(np.exp(x),axis=0)

# x=np.array([2.0,1.0,0.1])
# outputs=softmax(x)
# print('softmax numpy: ',outputs)

# x = torch.Tensor([2.0,1.0,0.1])
# outputs=torch.softmax(x,dim=0)
# print(outputs)

########################################################################

# def cross_entropy(actual,predicted):
#     loss = -np.sum(actual*np.log(predicted))
#     return loss

# Y=np.array([1,0,0])
# Y_pred_good=np.array([0.7,0.2,0.1])
# Y_pred_bad=np.array([0.1,0.3,0.6])
# l1=cross_entropy(Y,Y_pred_good)
# l2=cross_entropy(Y,Y_pred_bad)
# print(f'Loss1 numpy: {l1:.4f}')
# print(f'Loss2 numpy: {l2:.4f}')

# loss = nn.CrossEntropyLoss()

# Y=torch.tensor([2,0,1])
# #nsamples * nclasses =1x3
# Y_pred_good=torch.tensor([[ 0.1 , 1.0 , 2.1 ],[ 2.0 , 1.0 , 0.1 ],[ 0.1 , 3.0 , 0.1]])
# Y_pred_bad=torch.tensor([[ 2.1 , 1.0 , 0.1 ],[ 0.1 , 1.0 , 2.1 ], [ 0.1 , 3.0 , 0.1]])

# l1=loss(Y_pred_good,Y)
# l2=loss(Y_pred_bad,Y)
# print(f'Loss1 numpy: {l1:.4f}')
# print(f'Loss2 numpy: {l2:.4f}')

# _1,predictions1 = torch.max(Y_pred_good,1)
# _2,predictions2 = torch.max(Y_pred_bad,1)
# print(_1,predictions1)
# print(_2,predictions2)

###############################################

# # Multiclass problem
# class NeuralNet2(nn.Module):
#     def __init__(self,input_size, hidden_size, num_classes):
#         super(NeuralNet2,self).__init__()
#         self.linear1=nn.Linear(input_size,hidden_size)
#         self.relu=nn.ReLU()
#         self.linear2=nn.Linear(hidden_size,num_classes)
    
#     def forward(self,x):
#         out=self.linear1(x)
#         out=self.relu(out)
#         out=self.linear2(out)
#         #nosoftmax
#         return out
    
# model=NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
# criterion=nn.CrossEntropyLoss() #applies softmax

#Binary Classification
class NeuralNet1(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(NeuralNet1,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(hidden_size,1)
    
    def forward(self,x):
        out=self.linear1(x)
        out=self.relu(out)
        out=self.linear2(out)
        # sigmoid at end
        y_pred=torch.sigmoid(out)
        return y_pred
    
model=NeuralNet1(input_size=28*28, hidden_size=5)
criterion=nn.BCELoss() #applies softmax