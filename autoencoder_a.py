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

batch_size = 1000 
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
validloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
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

autoencoder = Autoencoder(512, dropout_p= 0.1)

criterion = nn.MSELoss()
optimizer = optim.SGD(autoencoder.parameters() , lr = 0.1) #otras pruebas lr = 0.01 / 0.001



#Ajuste del modelo
epochs = 50
train_losses = []
valid_losses = []
test_losses = []
for epoch in range(epochs):
    train_loss = 0.0
    valid_loss = 0.0
    test_loss = 0.0

    #ENTRENAMIENTO
    autoencoder.train()
    for data in trainloader:
        images, _ = data
        optimizer.zero_grad()
        outputs = autoencoder(images.view(images.size(0), -1))
        loss = criterion(outputs, images.view(images.size(0), -1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    #Validacion
    autoencoder.eval()
    for data in validloader:
        images, _ = data
        outputs = autoencoder(images.view(images.size(0), -1))
        loss = criterion(outputs, images.view(images.size(0), -1))
        valid_loss += loss.item() * images.size(0)
    
    #Testeo
    autoencoder.eval()
    for data in testloader:
        images, _ = data
        outputs = autoencoder(images.view(images.size(0), -1))
        loss = criterion(outputs, images.view(images.size(0), -1))
        test_loss += loss.item() * images.size(0)
    
    train_loss = train_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(validloader.sampler)
    test_loss = test_loss/len(testloader.sampler)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    test_losses.append(test_loss)


    print(f'Epoch: {epoch + 1}/{epochs}\t Training Loss: {train_loss:.6f}\t Validation Loss: {valid_loss:.6f}\t Test Loss: {test_loss:.6f}')

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training, Validation, and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Probrando el modelo ya entrenado
autoencoder.eval()
with torch.no_grad():
    dataiter = iter(testloader)
    images, _ = next(dataiter)

    original_images = images.view(-1,28,28)[:10]
    plt.figure(figsize=(20,4))
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(original_images[i].numpy(), cmap = 'gray')
        plt.title('Original')
        plt.axis('off')
    
    outputs = autoencoder(images.view(images.size(0),-1))
    reconstructed_images = outputs.view(-1,28,28)[:10]
    for i in range(10):
        plt.subplot(2,10,i+11)
        plt.imshow(reconstructed_images[i].numpy(), cmap = 'gray')
        plt.title('Reconstruida')
        plt.axis('off')
    plt.show()
