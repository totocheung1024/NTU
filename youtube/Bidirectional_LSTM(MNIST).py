import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

device="cuda" if torch.cuda.is_available() else "cpu"

input_size=28 #Width
sequence_length=28#Height
num_layers=2
hidden_size=256
num_classes=10
learning_rate=3e-4
batch_size=64
num_epochs=2


class BRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers,num_classes):
        super(BRNN,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        #for batch_first=True , shape will be (batch_size, sequence_length, features).
        #for batch_first=False, shape will be ((sequence_length, batch_size, features))
        
        
        #self.fc=nn.Linear(hidden_size*2, num_classes)
        self.fc=nn.Linear(hidden_size*2*sequence_length, num_classes)
        self.flatten=nn.Flatten()
        
    def forward(self,x):
        h0=torch.zeros(self.num_layers*2, x.size(0),self.hidden_size).to(device)
        #h0= (num_layers* num_directions, batch_size, hidden_size)
        c0=torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        #co=(num_layers * num_directions, batch_size, hidden_size)
        
        out,_=self.lstm(x,(h0, c0))
        out=self.flatten(out)
        out=self.fc(out)
        
        #out,_=self.lstm(x,(h0, c0))
        #out=self.fc(out[:,-1,:])
        #
        #
        
        
        return out
    
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=False
)

#Original shape of the image: (1,28,28)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=False
)

train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
#(batcn_size,1,28,28)

test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True)

model=BRNN(input_size,hidden_size,num_layers,num_classes).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for batch_inx,(data,targets) in enumerate(tqdm(train_loader)):
        data=data.to(device=device).squeeze(1)
        #if data.shape be like (batch_size,3,28,28)
        #Then data = data.reshape(batch_size, 28, -1)
        # New shape: (batch_size, 28, 3*28)
        targets=targets.to(device=device)
        
        #Forward
        scores=model(data)
        loss=criterion(scores,targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        

def check_accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checkinng accuracy on test data")
    
    num_correct=0
    num_samples=0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device).squeeze(1)
            #(Squeeze means destory the channel of [1])
            y=y.to(device=device)
            
            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
            
        print(
            f"Got {num_correct} / {num_samples} with accuracy  \
              {float(num_correct)/float(num_samples)*100:.2f}"
        )
    model.train()
    
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
            
            
            
        
