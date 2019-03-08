from torch import nn


class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        self.fc1 = nn.Linear(z_size, conv_dim * 4 * 4 * 4)
        # 4x4 -> 8x8
        self.deconv1 = deconvolution_layer_helper(conv_dim * 4, conv_dim * 2, 4)
        # 8x8 -> 16x16
        self.deconv2 = deconvolution_layer_helper(conv_dim * 2, conv_dim, 4)
        # 16x16 -> 32x32
        self.deconv3 = deconvolution_layer_helper(conv_dim, 3, 4, batch_norm=False)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc1(x)
        x = x.view(batch_size, self.conv_dim * 4, 4, 4)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.tanh(self.deconv3(x))
        return x


def deconvolution_layer_helper(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []

    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)
    layers.append(deconv)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)
