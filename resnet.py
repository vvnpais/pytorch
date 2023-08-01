import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy


class block(nn.Module):
    def __init__(self,in_channels,out_channels,identity_downsample=None,stride=1):
        super(block,self).__init__()
        self.expansion=4
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv3=nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)
        self.relu=nn.ReLU()
        self.identity_downsample=identity_downsample

    def forward(self,x):
        identity=x

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn3(x)

        if self.identity_downsample is not None:
            identity=self.identity_downsample(identity)

        x+=identity
        x=self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self,block,layers,image_channels,num_classes):
        super(ResNet,self).__init__()
        self.conv1=nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.in_channels=64

        # Resnet layers
        self.layer1=self._make_layer(block,layers[0],out_channels=64,stride=1)
        self.layer2=self._make_layer(block,layers[1],out_channels=128,stride=2)
        self.layer3=self._make_layer(block,layers[2],out_channels=256,stride=2)
        self.layer4=self._make_layer(block,layers[3],out_channels=512,stride=2)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*4,num_classes)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc(x)
        return x

    def _make_layer(self,block,num_residual_blocks,out_channels,stride):
        identity_downsample=None
        layers=[]

        if stride!=1 or self.in_channels!=out_channels*4:
            identity_downsample=nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride=stride),nn.BatchNorm2d(out_channels*4))

        layers.append(block(self.in_channels,out_channels,identity_downsample,stride))
        self.in_channels=out_channels*4 #64*4

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)
    
def ResNet50(img_channels=3,num_classes=1000):
    return ResNet(block,[3,4,6,3],img_channels,num_classes)
def ResNet101(img_channels=3,num_classes=1000):
    return ResNet(block,[3,4,23,3],img_channels,num_classes)
def ResNet152(img_channels=3,num_classes=1000):
    return ResNet(block,[3,8,36,3],img_channels,num_classes)

synset=open('../kagglefiles/files/LOC_synset_mapping.txt')
syn=synset.read().split('\n')
while '' in syn:
    syn.remove('')
for i in range(len(syn)):
    # print(syn[i])
    ind=syn[i].index(' ')
    syn[i]=[syn[i][:ind],syn[i][ind+1:]]
syn={i:syn[i] for i in range(len(syn))}

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean=np.array([0.485 , 0.456 , 0.406])
std=np.array([0.229 , 0.224 , 0.225])
learning_rate=0.001

data_transforms={
    'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ,transforms.Normalize(mean,std)
    ]),
    'val':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
}

data_dir='/home/vivian/vv/imagenet_site'
sets=['train','val']

image_datasets={ x :datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
num_work=8
dataloaders={}
dataloaders['train']=torch.utils.data.DataLoader(image_datasets['train'],batch_size=256,shuffle=True,num_workers=min(num_work,os.cpu_count()))
dataloaders['val']=torch.utils.data.DataLoader(image_datasets['val'],batch_size=256,shuffle=False,num_workers=min(num_work,os.cpu_count()))

dataset_sizes={x:len(image_datasets[x]) for x in ['train','val']}
class_names=image_datasets['train'].classes


# to check if images are correct according to their labels. this snippet will give proper class names in output
# samples,labels=next(iter(dataloaders['train']))
# #print(samples.shape,labels.shape)
# #print(samples[0][0],labels[0])
# for i in range(9):
#    print(syn[labels[i].item()])
#    plt.subplot(3,3,1+i)
#    grid_img=torchvision.utils.make_grid(samples[i],nrow=3)
#    plt.imshow(grid_img.permute(1,2,0))
# plt.show()


#print(class_names)

model=ResNet152(num_classes=1000).to(device)

# to check format of output
# for s,l in dataloaders['train']:
#     s=s.to(device)
#     l=l.to(device)
#     print(l[:9])
#     x=model(s)
#     print(torch.max(x[:9],1))
#     break


FILE="resnet.pth"
continue_training=False
if continue_training:
    model.load_state_dict(torch.load(FILE))
# print(model.parameters())
# for i in model.parameters():
#     print(i.shape)
# a=0
# for param in model.parameters():
#     t=param.view(-1)
#     a+=t.shape[0]
# print(a)
# print(f'{a//1000000},{(a//1000)%1000:03d},{a%1000:03d}')
for param in model.parameters():
    param.requires_grad=True
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
scheduler=lr_scheduler.StepLR(optimizer,step_size=dataset_sizes['train']/256,gamma=0.9)

since=time.time()
best_model_wts=copy.deepcopy(model.state_dict())
best_acc=0.0
epoch=1
optimize=True
while optimize:
    print(f'Epoch {epoch}')
    print('-'*10)
    for phase in sets:
        if phase=='train':
            model.train() #set model to training mode
        else:
            model.eval() #set model to evaluation mode
        running_loss=0.0
        running_corrects=0

        for inputs,labels in dataloaders[phase]:
            inputs=inputs.to(device)
            labels=labels.to(device)

            with torch.set_grad_enabled(phase=='train'):
                outputs=model(inputs)
                _,preds=torch.max(outputs,1)
                loss=criterion(outputs,labels)
                print(f'{loss.item():03.9f}',end='\r')

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            running_loss+=loss.item()*inputs.size(0)
            running_corrects+=torch.sum(preds==labels.data)
        if phase=='train':
            scheduler.step()
        epoch_loss=running_loss/dataset_sizes[phase]
        epoch_acc=running_corrects.double()/dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc:{epoch_acc:.4f}')

        if phase=='val' and epoch_acc>best_acc:
            torch.save(model.state_dict(),FILE)
            best_acc=epoch_acc
            best_model_wts=copy.deepcopy(model.state_dict())
            if best_acc>0.9551:
                optimize=False
    print()
