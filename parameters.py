batch_size = 64
img_size = 64
d_conv_dim = 64
g_conv_dim = 64
z_size = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.99

d_kernel_size = 4
d_stride = 2
d_padding = 1
d_conv_layers = 4

g_kernel_size = 4
g_stride = 2
g_padding = 1
g_conv_layers = 4

assert (img_size - d_kernel_size + 2 * d_padding) % 2 == 0
assert (2 * img_size / (img_size - d_kernel_size + 2 * d_padding + 2)) % 1 == 0
