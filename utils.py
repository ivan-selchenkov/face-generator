import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

import numpy as np

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


