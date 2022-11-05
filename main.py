import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_lsun_dataloader
from models import Generator, Discriminator
from training import Trainer
import matplotlib.pyplot as plt
import os

data_loader, _ = get_mnist_dataloaders(batch_size=64)
img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)

# Initialize optimizers
lr = 5e-4
betas = (.9, .99)
G_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
D_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)

# Train model
epochs = 100
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)
for key in trainer.losses:
  plt.plot(trainer.losses[key])
  plt.title(key)
  plt.savefig(os.path.join("./",key+".png"))
  plt.clf()

# Save models
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
