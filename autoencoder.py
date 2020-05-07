import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

device = ('cuda' if torch.cuda.is_available() else 'cpu')


dataset = datasets.load_iris()

x = dataset.data
y = dataset.target

print(type(x), type(y))
print(x.shape)

x = MinMaxScaler().fit_transform(x)

X = torch.from_numpy(x).to(device)


class Autoencoder(nn.Module):
    def __init__(self, x_features, out_features):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_features, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128 ,64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, out_features)
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(out_features),
            nn.Linear(out_features, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, x_features)
        )
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x 

if __name__ == '__main__':

    model = Autoencoder(4, 2).double().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    model.train()

    for epoch in range(0,5000):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()

        if epoch%500 == 0:
            print(f'Epoch {epoch}: \t Loss: {loss.item():.4g}')
        if epoch%1000 == 0:
            torch.save(model.state_dict(), './autoenc_iris.pt')

    with torch.no_grad():
        encoded = model.encoder(X)
        decoded = model.decoder(encoded)
        enc = encoded.cpu().detach().numpy()
        df = pd.DataFrame(columns=['1st_dim', '2nd_dim', 'target'])
        df['1st_dim'] = enc[:,0]
        df['2nd_dim'] = enc[:,1]
        df['target'] =  y

        df.to_csv('iris_reduction.csv')


        

