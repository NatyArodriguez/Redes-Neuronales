import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# carga de datos
transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
validset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)

class CustomDataset(Dataset):
    def __init__(self,dataset):
        self.dataset=dataset
    def __len__(self):
        return len(self.dataset)    
    def __getitem__(self,i):
        image, label =self.dataset[i]
        inputs = image
        output = image
        return inputs,output

trainset = CustomDataset(trainset)
validset = CustomDataset(validset)

class_names = ['T-shirt',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle Boot']

## Definicion de la red

class autoencoder(nn.Module):
    def __init__(self, n=64, p=0.2):
        super().__init__()
        self.n = n
        self.p = p
        #self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            #Primera capa Conv
            nn.Conv2d(1,16,kernel_size=3),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.MaxPool2d(2,2),
            #Segunda capa Conv
            nn.Conv2d(16,32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.MaxPool2d(2,2),
            #Lineal
            nn.Flatten(),
            nn.Linear(32*5*5,self.n),
            nn.ReLU(),
            nn.Dropout(self.p)
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n,32*5*5),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Unflatten(1,(32,5,5)),
            nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,output_padding=1),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,output_padding=1),
            nn.Sigmoid(),
            nn.Dropout(self.p)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
n=64
p=0.2

model = autoencoder(64,0.2)
autoencoder_conv = autoencoder(n=n,p=p)
#print(model)
########## Para ayudar en la visualizacion de las imagenes ######
def batch(x):
    return x.unsqueeze(0) #(28,28) -> (1,28,28)

def unbatch(x):
    return x.squeeze().detach().cpu().numpy() #(1,28,28) -> (28,28)
#############################################################

'''
figure = plt.figure()
rows, cols = 4,2
i = 0
for row in range(1, rows +1):
    j = torch.randint(len(trainset),size=(1,)).item()
    i += 1
    image,_ = trainset[j]
    figure.add_subplot(rows,cols,i)
    if row==1:
        plt.title('Original')
    plt.axis('off')
    plt.imshow(unbatch(image),cmap='Greys_r')
    i +=1
    figure.add_subplot(rows,cols,i)
    if row == 1:
        plt.title('Predicha')
    plt.axis('off')
    image_pred = unbatch(model(batch(image)))
    plt.imshow(image_pred,cmap='Greys_r')
plt.show()
'''

# Entrenando el autoencoder
def trainloop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss = 0.0
    model.to(device)
    for batch,(X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch = loss.item()
        sum_loss += loss_batch
        if batch % 100 == 0:
            current = batch*len(X)
            print(f"@trainloop batch={batch:>5d} loss={loss:>7f}  muestras-procesadas:[{current:>5d}/{size:>5d}]")
    avg_loss = sum_loss/num_batches
    return avg_loss

def validloop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss = 0.0
    sum_correct = 0.0
    model.to(device)
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss_batch = loss_fn(pred,y).item()
            sum_loss += loss_batch
            #sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item() 
    avg_loss = sum_loss/num_batches
    print(f'@validloop avg_loss={avg_loss:8f}')
    #frac_correct = sum_correct/size
    #print(f"Test Error: \n Accuracy: {(100*frac_correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return avg_loss


batchs = 100

trainloader = DataLoader(trainset, batch_size=batchs,shuffle=True)
validloader = DataLoader(validset, batch_size=batchs,shuffle=True)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 15

list_avg_train_loss = []
list_avg_valid_train_loss =[]
list_avg_valid_loss = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

for epochs in range(num_epochs):
    print(f'Epoch {epochs +1} \n ......................................')
    avg_train_loss = trainloop(trainloader,model,loss_fn,optimizer)
    avg_valid_train_loss = validloop(trainloader,model,loss_fn)
    avg_valid_loss = validloop(validloader,model,loss_fn)
    list_avg_train_loss.append(avg_train_loss)
    list_avg_valid_train_loss.append(avg_valid_train_loss)
    list_avg_valid_loss.append(avg_valid_loss)
    #print(f'Epoch: {epochs + 1}/{num_epochs}\t Training Loss Incorrecto: {avg_train_loss:.6f}\t Training Loss: {avg_valid_train_loss:.6f}\t Validation Loss: {avg_valid_loss:.6f}')


plt.xlabel('epoch')
plt.ylabel('loss')
#plt.plot(list(range(1,len(list_avg_train_loss)+1)),list_avg_train_loss,label="train",linestyle='-.',c='green')
plt.plot(list(range(1,len(list_avg_valid_loss)+1)),list_avg_valid_train_loss,label="valid-train",linestyle='-',c='magenta')
plt.plot(list(range(1,len(list_avg_valid_loss)+1)),list_avg_valid_loss,label="valid",linestyle='--',c='blue')
plt.title('')
plt.legend()
plt.show()


figure = plt.figure()
rows,cols = 3,2
i = 0
model.eval()
for row in range(1,rows+1):
    j = torch.randint(len(trainset),size=(1,)).item()
    i += 1
    image,flatten_imagen = trainset[j]
    figure.add_subplot(rows,cols,i)
    if row == 1:
        plt.title('Original')
    plt.axis('off')
    plt.imshow(unbatch(image),cmap='Greys_r')
    i +=1
    figure.add_subplot(rows,cols,i)
    if row ==1:
        plt.title('Predicha')
    plt.axis('off')
    pred = unbatch(model(batch(image)))
    plt.imshow(pred,cmap = 'Greys_r')
plt.show()



### PARTE 3 #####
########## clasificador ########
class clasificador(nn.Module):
    def __init__(self,autoencoder_conv=None,copy_encoder=True,n=64,p=0.2):
        super().__init__()
        if autoencoder_conv is None:
            print('Creating encoder')
            self.n = n
            self.p = p
            self.lei = 5
            self.ldo = 5
            self.encoder = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.Dropout(self.p),
                nn.MaxPool2d(2,2),
                nn.Conv2d(16,32,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.Dropout(self.p),
                nn.MaxPool2d(2,2),
                nn.Flatten(),
                nn.Linear(32*self.lei*self.lei,self.n),
                nn.ReLU(),
                nn.Dropout(self.p)
            )
        else:
            self.n = autoencoder_conv.n
            self.p =autoencoder_conv.p
            if copy_encoder:
                print('copyng provided encoder')
                self.encoder = copy.deepcopy(autoencoder_conv.encoder)
            else:
                print('Using provided encoder')
                self.encoder = autoencoder_conv.encoder
            
            self.clasificador = nn.Sequential(
                nn.Linear(self.n,10),
                nn.ReLU(),
                nn.Dropout(self.p)
            )
    def forward(self,x):
        x = self.encoder(x)
        x = self.clasificador(x)
        return x

clasificador_conv = clasificador(autoencoder_conv=autoencoder_conv)
model = clasificador_conv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

#para el entrenamiento del clasificador
def trainloop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss = 0.0
    sum_correct = 0.0
    model.to(device)
    for batch,(X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch = loss.item()
        sum_loss += loss_batch
        sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            current = batch*len(X)
            print(f"@trainloop batch={batch:>5d} loss={loss:>7f}  muestras-procesadas:[{current:>5d}/{size:>5d}]")
    avg_loss = sum_loss/num_batches
    correct = sum_correct/size
    return avg_loss, correct

def validloop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss = 0.0
    sum_correct = 0.0
    model.to(device)
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss_batch = loss_fn(pred,y).item()
            sum_loss += loss_batch
            sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item() 
    avg_loss = sum_loss/num_batches
    #print(f'@validloop avg_loss={avg_loss:8f}')
    frac_correct = sum_correct/size
    print(f"@validloop Accurary {(100*frac_correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return avg_loss,frac_correct

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,eps=1e-08,weight_decay=0,amsgrad=False)
num_epochs = 15

trainset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
validset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
trainloader = DataLoader(trainset, batch_size=batchs,shuffle=True)
validloader = DataLoader(validset, batch_size=batchs,shuffle=True)

list_avg_train_loss = []
list_avg_valid_train_loss =[]
list_avg_valid_loss = []
list_acc_train = []
list_acc_valid_train = []
list_acc_valid = []

for epochs in range(num_epochs):
    print(f'Epoch {epochs +1} \n ...................................................................')
    avg_train_loss, avg_acc_train = trainloop(trainloader,model,loss_fn,optimizer)
    avg_valid_train_loss, avg_acc_valid_train = validloop(trainloader,model,loss_fn)
    avg_valid_loss, avg_acc_valid = validloop(validloader,model,loss_fn)
    list_avg_train_loss.append(avg_train_loss)
    list_avg_valid_train_loss.append(avg_valid_train_loss)
    list_avg_valid_loss.append(avg_valid_loss)
    list_acc_train.append(avg_acc_train)
    list_acc_valid_train.append(avg_acc_valid_train)
    list_acc_valid.append(avg_acc_valid)
 
plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
plt.plot(range(1,num_epochs+1),list_avg_train_loss, label='Train Loss', color='r')
plt.plot(range(1,num_epochs+1), list_avg_valid_train_loss, label = 'Valid Train Loss', color='green')
plt.plot(range(1,num_epochs+1), list_avg_valid_loss, label= 'Valid Loss', color='magenta')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1,num_epochs+1),list_acc_train, label='Train Loss', color='r', linestyle = '-.')
plt.plot(range(1,num_epochs+1), list_acc_valid_train, label = 'Valid Train Loss', color='green', linestyle = '--')
plt.plot(range(1,num_epochs+1), list_acc_valid, label= 'Valid Loss', color='magenta',linestyle = ':')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


y_pred = []
y_true = []

size = len(validloader.dataset)
for batch,(inputs, labels) in enumerate(validloader):
    outputs = model(inputs)
    outputs = (torch.max(torch.exp(outputs),1)[1]).data.cpu().numpy()
    y_pred.extend(outputs)

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)
    if batch %10 == 0:
        current = batch*len(inputs)
        print(f'batch={batch:>5d} muestras procesadas:[{current:>5d}/{size:>5d}')

classes = ('T-shirt',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle Boot')

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:,None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sns.heatmap(df_cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
