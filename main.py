from utils import get_dataloader, display_images

batch_size = 64
img_size = 32

loader = get_dataloader(batch_size, img_size)

display_images(loader)

