from torch import nn

from parameters import d_conv_dim, img_size, d_stride, d_padding, d_kernel_size, d_conv_layers


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = generate_convolution_layers(d_conv_layers)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        batch_size = x.size(0)

        for i in range(d_conv_layers - 1):
            x = self.leaky_relu(self.layers[i])

        x = x.view(batch_size, -1)

        return self.layers[d_conv_layers - 1](x)


def convolution_layer_helper(in_channels, out_channels, batch_norm=True):
    layers = []

    conv = nn.Conv2d(in_channels, out_channels, d_kernel_size, d_stride, d_padding, bias=not batch_norm)

    layers.append(conv)

    if batch_norm:
        batch_norm = nn.BatchNorm2d(out_channels)
        layers.append(batch_norm)

    return nn.Sequential(*layers)


def generate_convolution_layers(conv_layers_number):
    layers = []

    layers.append(convolution_layer_helper(3, d_conv_dim, batch_norm=False))

    for i in range(conv_layers_number):
        in_channels = d_conv_dim * (2 ** i)
        out_channels = d_conv_dim * (2 ** (i + 1))
        layers.append(convolution_layer_helper(in_channels, out_channels))

    divider = 2 * img_size / (img_size - d_kernel_size + 2 * d_padding + 2)
    after_conv_size = img_size / (divider ** conv_layers_number)

    layers.append(nn.Linear(d_conv_dim * (2 ** (conv_layers_number + 1)) * after_conv_size * after_conv_size, 1))

    return layers


D = Discriminator()
