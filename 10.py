# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model Evaluation
# GPU Support
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size=784 #28*28
hidden_size =100
num_classes=10
num_epochs=6
batch_size=100
learning_rate=0.01

train_dataset=torchvision.datasets.MNIST(root='',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=torchvision.datasets.MNIST(root='',train=False,transform=transforms.ToTensor() )
train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False)

examples=iter(train_loader)
samples,labels=next(examples) 
print(samples,labels)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')
# plt.show()

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        return out
    
model=NeuralNet(input_size,hidden_size,num_classes)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
# FILE="model.pth"
# model.load_state_dict(torch.load(FILE))
# for i in model.parameters():
#     print(i)
# torch.save(model.state_dict(),FILE)
# exit()


# training loop
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        # 100,1,28,28
        # 100,784
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)

        # forward
        outputs=model(images)
        loss=criterion(outputs,labels)

        # back
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss={loss.item():.4f}')

# test
with torch.no_grad():
    n_correct=0
    n_samples=0
    for images,labels in test_loader:
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        outputs=model(images)
        # print(outputs.shape)


        # value,index
        _,predictions=torch.max(outputs,1)
        n_samples+=labels.shape[0]
        n_correct+=(predictions==labels).sum().item()
    
    acc=100.0*n_correct/n_samples
    print(f'accuracy: {acc}')