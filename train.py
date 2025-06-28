import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from VAE import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ! Hyperparameters
input_dim = 28*28
h_dim = 200
z_dim = 20 
num_epochs = 10
batch_size = 32
lr_rate = 3e-5

# ! Dataset
dataset = datasets.MNIST(root="dataset/" , train=True , transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset , batch_size=batch_size , shuffle=True)
model = VariationalAutoEncoder(input_dim , h_dim , z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters() , lr=lr_rate)

loss_fn = nn.BCELoss(reduction="sum")

for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader))
    for i , (x , _ ) in loop :
        # ? Falttening the Image
        x = x.to(device).view(x.shape[0] , input_dim)

        x_reconstruction , mu , sigma = model(x)

        reconstruction_loss = loss_fn(x_reconstruction , x)
        kl_div = -torch.sum(1+ torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))


        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss = loss.item())


torch.save(model.state_dict() , "vae1.pth")
