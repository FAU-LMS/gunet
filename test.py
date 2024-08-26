import time
import torch
from torchvision import transforms
import imageio

from network import GUNet
import matplotlib.pyplot as plt

def cuda(*args):
    return (item.to(0) for item in args)

def test(gpu=True):
    tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    np_swir = imageio.imread('test_data/swir.png')/255
    np_vis = imageio.imread('test_data/vis.png')/255
    lr_swir, hr_vis = tensor(np_swir).float(), tensor(np_vis).float()
    lr_swir, hr_vis = lr_swir[None, ...], hr_vis[None, ...]
    if gpu:
        lr_swir, hr_vis = cuda(lr_swir, hr_vis)

    img = GUNet.run(lr_swir, hr_vis)

    img = img.detach().cpu().numpy()[0, 0]

    plt.subplot(131)
    plt.imshow(np_vis, cmap='gray')
    plt.subplot(132)
    plt.imshow(np_swir, cmap='gray')
    plt.subplot(133)
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    test()
