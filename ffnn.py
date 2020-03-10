import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = torchvision.datasets.MNIST(root='./data', train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)


test_data = torchvision.datasets.MNIST(root='./data', train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = 64,
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = 64,
                                            shuffle=True)


examples = iter(train_loader)

samples, labels = examples.next()



class NeuralNetwork(nn.Module):
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_layer_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer_size, 50)
        self.l3 = nn.Linear(50, output_layer_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        
        return out


if __name__ == '__main__':
    loss = 0
    print(f'Training on: {device}')
    model = NeuralNetwork(784, 200, 10).cuda()
    num_of_epochs = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
    start = time.time()
    for epochs in range(num_of_epochs):
        
        for i, (images, labels) in enumerate(train_loader):
            
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)


            outputs = model(images)
            loss = criterion(outputs, labels)

            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if (i+1) % 64 == 0:
                print(f'epoch {epochs+1}: / {num_of_epochs}, step  {i}/{len(train_loader)} loss: {loss.item():.4f}')

    end = time.time()

    print(end-start)
    
    with torch.no_grad():
      
        n_correct = 0
        n_samples = 0
        
        for images, labels in test_loader:

            images= images.reshape(-1, 784).to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            
            n_correct += (predictions == labels).sum().item()
            
        
        acc = 100.0 * n_correct / n_samples
        


        print(f'Accuracy: {acc}')
        print(f'Loss: {loss}')

        