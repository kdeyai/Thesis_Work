import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from torchvision.transforms.autoaugment import InterpolationMode
from torchvision import transforms
from segmentation_models_pytorch.losses import FocalLoss
from torchmetrics.classification import  BinarySpecificity, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryAccuracy
import numpy as np
from torch.autograd import Variable
import torchvision

    

class MLPHead(nn.Module):
    def __init__(self, inch):
     super(MLPHead,self).__init__()
     self.l1 = nn.Linear(inch, 512, device= 'cuda')
     self.l2 = nn.Linear(512, 256, device = 'cuda')
     self.l3 = nn.Linear(256, 128, device = 'cuda')
     self.l4 = nn.Linear(128, 64, device = 'cuda')
     self.l5 = nn.Linear(64, 10, device = 'cuda')
    
    def forward(self, x):
      x = self.l1(x)
      x = self.l2(x)
      x = self.l3(x)
      x = self.l4(x)
      x = self.l5(x)

      return x

class SelfAttention(nn.Module):
   def __init__(self, key_dim = 16, patch_size = 16):
     super(SelfAttention,self).__init__()
     self.Wq = torch.nn.Parameter(torch.randn(patch_size, key_dim), requires_grad=True).to(device)
     self.Wk = torch.nn.Parameter(torch.randn(patch_size, key_dim), requires_grad=True).to(device)
     self.Wv = torch.nn.Parameter(torch.randn(patch_size, key_dim), requires_grad=True).to(device)

   def forward(self, x):

     Q = torch.matmul(x, self.Wq)
     K = torch.matmul(x, self.Wk)
     V = torch.matmul(x, self.Wv)

     z = torch.matmul(F.softmax(torch.div(torch.bmm(Q, K.permute(0,2,1)),torch.tensor(3))),V)
     return z

class NeuralNetwork(nn.Module):
    def __init__(self, inch, outch):
     super(NeuralNetwork,self).__init__()
     self.l1 = nn.Linear(inch, outch,device= 'cuda')
     self.r1 = nn.ReLU()
     self.l2 = nn.Linear(inch, outch,device ='cuda')
     self.r2 = nn.ReLU()
    
    def forward(self, x):
      x1 = self.l1(x)
      x1 = self.r1(x1)
      x2 = self.l2(x1)
      x2 = self.r2(x2)

      return x2


class Encoder(nn.Module):
    def __init__(self,key_dim, patch_size):
     super(Encoder,self).__init__()
     self.attn = SelfAttention(key_dim,patch_size)
     self.nn = []
     for i in range(0,64):
       self.nn.append(NeuralNetwork(key_dim,key_dim))
    
    def forward(self, x):
      y = self.attn(x)
      x= x + y
      ln = nn.LayerNorm([64,16])
      ln.cuda()
      x = ln(x)
      y = x
      n = []
      for i in range(0,64):
        n.append(self.nn[i](x[:,i]))
      x = torch.stack(n, dim = 2).permute(0,2,1)
      x = x + y
      x = ln(x)
      
      return x

class PositionalEncoding(nn.Module):
    def __init__(self, patch_size = 16):
        super(PositionalEncoding, self).__init__()

        
    def forward(self, x):
        x = x.view((-1,64,patch_size))
        y = x
        for i in range(x.size()[1]):
          for j in range(patch_size):
            if(j % 2 == 0):
              div = i / torch.pow(torch.tensor(10000),torch.tensor(j / patch_size*2))
              x[:,i,j] = torch.sin(div)
            else:
              div = i / torch.pow(torch.tensor(10000),torch.tensor(j / patch_size*2))
              x[:,i,j] = torch.cos(div)
        x1 = y + x
        return x1

class ViTModel(nn.Module):
   def __init__(self, patch_size = 16,key_dim = 16):
     super(ViTModel,self).__init__()
     self.pos = PositionalEncoding(patch_size)
     self.enc = Encoder(key_dim, patch_size)
     self.mlp = MLPHead(1024)
   def forward(self, x):
    
    x = self.pos(x)
    x = self.enc(x)
    x = torch.flatten(x, start_dim = 1)
    x = self.mlp(x)

    return x  




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size_train = 128
batch_size_test = 128
train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('/home/kaushik/Desktop/Kaushik_MTP/transformer_scratch/', train=True, download=True,
                          transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                              (0.1307,), (0.3081,)),
                            torchvision.transforms.Resize((32,32)),
                            torchvision.transforms.Lambda(lambda x: torch.flatten(x))  
                          ])),
batch_size=batch_size_train)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('/home/kaushik/Desktop/Kaushik_MTP/transformer_scratch/', train=False, download=True,
                          transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                              (0.1307,), (0.3081,)),
                            torchvision.transforms.Resize((32,32)),
                            torchvision.transforms.Lambda(lambda x: torch.flatten(x))    
                          ])),
batch_size=batch_size_test)
patch_size = 16     # patch dimension  16 x 16
key_dim = 16        # key dimension of the self attention layer 16
model1 = ViTModel(key_dim,patch_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model1.parameters(), lr=0.0000001,weight_decay=0.000001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience =2,factor=0.01, threshold=1e-15, verbose=True)
epochs = 10


count_train = len(train_loader.dataset)
count_test = len(test_loader.dataset)
train_loss = []
val_loss = []
train_acc = []
val_acc = []
print('training started')

for epoch in range(epochs):
    start_time = time.time()
    model1.train()
    trloss = 0
    valloss = 0
    num_batches = 0
    correct = 0
    print('epochs {}'.format(epoch+1))
    for batch in train_loader: 
        data = batch[0]
        targets = batch[1] 
        optimizer.zero_grad() 
        data = data.to(device)
        targets = targets.to(device)
        outputs = model1(data)
        outputs = F.softmax(outputs, dim=1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        trloss += loss.item()
        num_batches +=1
        correct += (torch.argmax(outputs,dim =1) == targets).float().sum().item()
    trloss /= num_batches
    correct/= count_train
    print('train loss: {:.4f} , train acc:{:.4f}'.format(trloss,correct))
    train_loss.append(trloss)
    train_acc.append(correct)
    num_batches = 0
    correct = 0
    model1.eval()
    with torch.no_grad():
        for batch in test_loader: 
            data = batch[0]
            targets = batch[1] 
            data = data.to(device)
            targets = targets.to(device)
            outputs = model1(data)
            outputs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)
            valloss += loss.item()
            num_batches +=1
            correct += (torch.argmax(outputs,dim =1) == targets).float().sum().item()
    valloss /= num_batches
    correct/= count_test
    scheduler.step(valloss)
    print('val loss: {:.4f}, val acc:{:.4f}'.format(valloss,correct))
    val_loss.append(valloss)
    val_acc.append(correct)
    end_time = time.time()
    print('Time taken:{:.4f} minutes'.format((end_time - start_time)/60))
    print()
    
print('completed... training and testing')