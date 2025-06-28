import torch
from torch import nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim=200, z_dim=20):
        super().__init__()

        # ! For Encoder
        self.img_2_hid = nn.Linear(input_dim, hid_dim)

        self.hid_2_mu = nn.Linear(hid_dim, z_dim)
        self.hid_2_sigma = nn.Linear(hid_dim, z_dim)

        # ! For Decoder 
        self.z_2_hid = nn.Linear(z_dim , hid_dim)
        self.hid_2_img = nn.Linear(hid_dim , input_dim)

        # ! Extra 
        self.relu = nn.ReLU()

    def encode(self , x):
        h = self.relu(self.img_2_hid(x))
        mu , sigma = self.hid_2_mu(h) ,self.hid_2_sigma(h )

        return mu , sigma
    
    def decode(self , z) :
        h = self.relu(self.z_2_hid(z))
        return torch.sigmoid(self.hid_2_img(h))
    
    def forward(self , x):
        mu , sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)

        z_repametrized = mu + sigma * epsilon 

        x_reconstructed = self.decode(z_repametrized)

        return x_reconstructed , mu , sigma
    

if __name__ == "__main__" :
    x = torch.randn(4 , 28*28)
    vae = VariationalAutoEncoder(28*28 )
    x_re , mu , sigma = vae(x)
    
    print(x_re.shape)
    print(mu.shape)
    print(sigma.shape)


