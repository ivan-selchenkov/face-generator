import torch.optim as optim

from dataloader import loader
from discriminator import Discriminator
from generator import Generator
from parameters import d_conv_dim, z_size, g_conv_dim, lr, beta1, beta2
from train import train
from utils import display_images, weights_init_normal, gpu_check, load_samples, view_samples

display_images(loader)


D = Discriminator()
G = Generator(z_size=z_size, conv_dim=g_conv_dim)

D.apply(weights_init_normal)
G.apply(weights_init_normal)

train_on_gpu = gpu_check()

if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')


d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

n_epochs = 10

losses = train(D, d_optimizer, G, g_optimizer, n_epochs=n_epochs)

samples = load_samples()

_ = view_samples(-1, samples)