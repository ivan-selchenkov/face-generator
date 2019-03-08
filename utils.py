import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import pickle as pkl

import numpy as np

criterion = nn.BCEWithLogitsLoss()

def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder('./' + data_dir, transform)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return loader


def display_images(dataloader):
    dataiter = iter(dataloader)
    images, _ = dataiter.next()

    fig = plt.figure(figsize=(20, 4))
    plot_size=20
    for idx in np.arange(plot_size):
        fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
        npimg = images[idx].numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

def scale(x, feature_range=(-1, 1)):
    min, max = feature_range
    return x * (max - min) - max

def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network
    """

    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        m.weight.data.normal_(mean=0, std=0.02)

def gpu_check():
    return torch.cuda.is_available()

def real_loss(D_out):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    batch_size = D_out.size(0)

    labels = torch.ones(batch_size)

    if gpu_check():
        labels = labels.cuda()

    return criterion(D_out.squeeze(), labels)

def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    batch_size = D_out.size(0)

    labels = torch.zeros(batch_size)

    if gpu_check():
        labels = labels.cuda()

    return criterion(D_out.squeeze(), labels)


def generate_fixed_z(size):
    fixed_z = torch.ones(size).uniform_(-1, 1)

    if gpu_check():
        fixed_z = fixed_z.cuda()

    return fixed_z


def plot_losses(losses):
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.show()


def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))
    plt.show()


def dump_samples(samples):
    with open('train_samples_charm.pkl', 'wb') as f:
        pkl.dump(samples, f)


def load_samples():
    with open('train_samples_charm.pkl', 'rb') as f:
        samples = pkl.load(f)
    return samples

