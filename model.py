import torch
import torch.nn as nn
import torch.nn.functional as F


number_of_classes = 11
in_class_min = 0
in_class_max = 4
latent_dim = 8


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        #for encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=2)
        self.fc11 = nn.Linear(in_features=3146, out_features=16)
        self.fc12 = nn.Linear(in_features=16, out_features=latent_dim)
        self.fc13 = nn.Linear(in_features=16, out_features=latent_dim)

        #for decoder
        self.fc21=nn.Linear(in_features=latent_dim+number_of_classes-1, out_features=16)
        self.fc22=nn.Linear(in_features=16, out_features=3136)
        self.conv3=nn.ConvTranspose2d(in_channels=64, out_channels=32, padding=1,output_padding=1,kernel_size=(3, 3),stride=2)
        self.conv4=nn.ConvTranspose2d(in_channels=32, out_channels=16, padding=1,output_padding=1,kernel_size=(3, 3), stride=2)
        self.conv5=nn.ConvTranspose2d(in_channels=16, out_channels=1, padding=15,output_padding=1,kernel_size=(3, 3), stride=2)

    def encoder(self, x, y):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(len(x),-1)
        x=torch.cat([x,y],dim=-1)
        x=F.relu(self.fc11(x))
        mu=self.fc12(x)
        var=self.fc13(x)
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)+0.0000000000001

        return mu + eps * std,mu,var

    def decoder(self, z,y):
        m=torch.cat([z,y],dim=-1)
        m = F.relu(self.fc21(m))
        m = F.relu(self.fc22(m))
        m = m.view(-1, 64,7, 7)
        m = F.relu(self.conv3(m))
        m = F.relu(self.conv4(m))
        recon_x = F.sigmoid(self.conv5(m))
        return recon_x

    def forward(self, x,y):
        z,mu,var= self.encoder(x,y)
        return self.decoder(z,y), mu, var,z

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x,size_average=False)
    KLD = 0.5 * torch.sum( mu.pow(2) + logvar.exp() - logvar - 1)
    return (BCE + KLD)/x.shape[0]

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=2)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(in_features=576, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=11)

    def forward(self, x):
        x=x.view(-1,1,28,28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x= self.conv2_drop(x)
        x=x.view(x.shape[0],-1)
        x=F.relu(self.fc1(x))
        x=F.softmax(self.fc2(x), dim=1)
        return x

