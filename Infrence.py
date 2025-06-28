import torch
from torchvision.utils import save_image
from torchvision import transforms
import torchvision.datasets as datasets

# from train import dataset
from VAE import VariationalAutoEncoder

model = VariationalAutoEncoder(28*28)
model.load_state_dict(torch.load("vae1.pth"))
dataset = datasets.MNIST(root="dataset/" , train=True , transform=transforms.ToTensor(), download=True)

images = []
idx = 0 
for x , y in dataset :
        if y== idx :
            images.append(x)
            idx+=1
        if idx == 10 :
            break
encodings = []
for d in range(10):
        with torch.no_grad():
            mu , sigma = model.encode(images[d].view(1 , 784))
            encodings.append((mu , sigma))

digit = 2
num_ex = 8

mu , sigma = encodings[digit]

for ex in range(num_ex):
    epsilon = torch.randn_like(sigma)

    #! Reparametrization Trick 
    z = mu + sigma * epsilon
    out = model.decode(z)
    out = out.view(-1 , 1 , 28 , 28)
    save_image(out , f"./GEN_IMG/gen_{digit}_{ex}.png")
    