import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,),)])

trainset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
testset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)

#print(len(trainset),len(testset))

indices = list(range(len(trainset)))
np.random.shuffle(indices)

split = int(np.floor(0.2 * len(trainset)))
train_indices, valid_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

batch_size = 1000 #Numero de muestras que se utilizaran en cada iteracion
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler)
validloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = valid_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True)

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

#Visualizamos algunas imagenes para controlar que la carga de datos sea correcta
figure =  plt.figure(figsize = (10,6))
cols, rows = 4,3

for i in range(1, cols *rows +1):
    j =np.random.randint(len(trainset))
    image , label = trainset[j]
    figure.add_subplot(rows, cols, i)
    plt.title(class_names[label])
    plt.axis('off')
    plt.imshow(image.squeeze(), cmap= 'Greys_r')
plt.show()

class Autoencoder(nn.Module):
    def __init__(self, n, dropout_p):
        super(Autoencoder, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, n),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n, 28*28),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_and_plot(hidden_size, ax_train, ax_test):
    autoencoder = Autoencoder(hidden_size, dropout_p = 0.1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(autoencoder.parameters(), lr=0.1)

    epochs = 50
    train_losses = []
    test_losses = []

    for epochs in range(epochs):
        train_loss = 0.0
        test_loss = 0.0

        autoencoder.train()
        for data in trainloader:
            images, _ = data
            optimizer.zero_grad()
            outputs = autoencoder(images.view(images.size(0),-1))
            loss = criterion(outputs, images.view(images.size(0),-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        autoencoder.eval()
        for data in testloader:
            images, _ = data
            outputs = autoencoder(images.view(images.size(0), -1))
            loss = criterion(outputs, images.view(images.size(0), -1))
            test_loss += loss.item()* images.size(0)
    
        train_loss = train_loss / len(trainloader.sampler)
        test_loss = test_loss / len(testloader.sampler)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    ax_train.plot(train_losses, label = f'Training Loss (L = {hidden_size})')
    ax_test.plot(test_losses, label = f'Test Loss (L = {hidden_size})')

fig_train, ax_train = plt.subplots()
fig_test, ax_test = plt.subplots()

hidden_sizes = [64, 128, 256, 512]

for hidden_size in hidden_sizes:
    train_and_plot(hidden_size, ax_train, ax_test)

ax_train.set_title('Training Loss')
ax_train.set_xlabel('Epochs')
ax_train.set_ylabel('MSE')
ax_train.legend()

ax_test.set_title('Test Loss')
ax_test.set_xlabel('Epochs')
ax_test.set_ylabel('MSE')
ax_test.legend()

plt.show()
