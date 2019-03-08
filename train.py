import pickle as pkl

from dataloader import loader
from parameters import z_size
from utils import gpu_check, generate_fixed_z, scale, real_loss, fake_loss, dump_samples
from torch import optim

train_on_gpu = gpu_check()

def train(D, d_optimizer: optim.Adam, G, g_optimizer: optim.Adam, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''

    if train_on_gpu:
        D.cuda()
        G.cuda()

    samples = []
    losses = []

    sample_size = 16

    fixed_z = generate_fixed_z((sample_size, z_size))

    for epoch in range(n_epochs):
        for batch_i, (real_images, _) in enumerate(loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            if train_on_gpu:
                real_images = real_images.cuda()

            d_optimizer.zero_grad()
            d_real_loss = real_loss( D(real_images) )

            fake_images = G(generate_fixed_z((batch_size, z_size)))
            d_fake_loss = fake_loss( D(fake_images) )

            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()

            d_optimizer.step()

            g_optimizer.zero_grad()
            fake_images = G(generate_fixed_z((batch_size, z_size)))

            g_loss = real_loss( D(fake_images) )
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, n_epochs, d_loss.item(), g_loss.item()))

        G.eval()  # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to training mode

        dump_samples(samples)

    # finally return losses
    return losses
