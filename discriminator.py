from torch import nn


class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()

        # 32x32 -> 16x16
        self.conv1 = convolution_layer_helper(3, conv_dim, 4, batch_norm=False)
        # 16x16 -> 8x8
        self.conv2 = convolution_layer_helper(conv_dim, conv_dim * 2, 4)
        # 8x8 -> 4x4
        self.conv3 = convolution_layer_helper(conv_dim * 2, conv_dim * 4, 4)

        self.fc1 = nn.Linear(conv_dim * 4 * 4 * 4, 1)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        batch_size = x.size(0)

        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))

        x = x.view(batch_size, -1)

        return self.fc1(x)


def convolution_layer_helper(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []

    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)

    layers.append(conv)

    if batch_norm:
        batch_norm = nn.BatchNorm2d(out_channels)
        layers.append(batch_norm)

    return nn.Sequential(*layers)
