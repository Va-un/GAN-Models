
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
class Discriminator(nn.Module):
  def __init__(self,img_dim):
    super().__init__()

    self.disc = nn.Sequential(
        nn.Linear(img_dim,128),
        nn.LeakyReLU(0.1),
        nn.Linear(128,1),
        nn.Sigmoid(),
    )

  def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
  def __init__(self,z_dim,img_dim): #z_dim = latent noise
     super().__init__()

     self.gen = nn.Sequential(
        nn.Linear(z_dim,256),
        nn.LeakyReLU(0.1),
        nn.Linear(256,img_dim),
        nn.Tanh(),
     )

  def forward(self,x):
         return self.gen(x)

#Hyperparameters
device =  "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
img_dim = 28*28*1
batch_size = 32
num_epochs = 100

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim,img_dim).to(device)
fixed_noise = torch.rand((batch_size,z_dim)).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]
)

dataset = datasets.MNIST(root = "/",transform = transforms,download = True )
loader = DataLoader(dataset,batch_size,shuffle = True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)

loader = DataLoader(dataset,batch_size,shuffle = True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/MNIST/fake")
writer_real = SummaryWriter(f"logs/MNIST/real")
step = 0



for epoch in range(num_epochs):
  for batch_idx,(real,_) in enumerate(loader):
    real = real.view(-1,784).to(device)
    batch_size = real.shape[0]

    ### train Discrimminator:
    noise = torch.rand(batch_size,z_dim).to(device)
    fake = gen(noise)
    disc_real = disc(real).view(-1)
    lossD_real = criterion(disc_real, torch.ones_like(disc_real))
    disc_fake = disc(fake).view(-1)
    lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    lossD = (lossD_real +lossD_fake) /2
    disc.zero_grad()
    lossD.backward(retain_graph = True)
    opt_disc.step()

    ### Train Generator
    output = disc(fake).view(-1)
    lossG = criterion(output,torch.ones_like(output))
    gen.zero_grad()
    lossG.backward()
    opt_gen.step()

    if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs
# Generate and visualize some fake images
with torch.no_grad():
    z = torch.randn(16, 100).to(device)
    fake_images = Generator(z).cpu()
    fake_images = fake_images.view(-1, 28, 28)

plt.figure(figsize=(4, 4))
for i in range(fake_images.size(0)):
    plt.subplot(4, 4, i + 1)
    plt.imshow(fake_images[i], cmap='gray')
    plt.axis('off')
plt.show()
