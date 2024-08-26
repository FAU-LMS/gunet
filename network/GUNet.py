import time
import torch
from torchvision import transforms
import torchvision.transforms as TT
import torch.nn.functional as F
import imageio
import cv2 as cv
import numpy as np
import os
import math
import pathlib
import matplotlib.pyplot as plt

from .network import GraphSuperResolutionNet

model_dict = {}

def cuda(*args):
    return (item.to(0) for item in args)

def load_model(model, path=None, gpu=True):
    print(f'[**] Loading network from {path}')
    if torch.cuda.is_available() and gpu:
        model.load_state_dict(torch.load(path, map_location='cuda'))
    else:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

def predict_patch(model, patch_lr_out, patch_hr_in):
    model.eval()
    with torch.no_grad():
        dict = model(patch_hr_in, patch_lr_out)
        img = dict['y_pred']
    return img

def run(low_res_out, high_res_in, gpu=True, reload_model=False, model_name='model/model.pt'):
    scale_factor = high_res_in.shape[2]/low_res_out.shape[2]
    hr_patch_size = 256
    lr_patch_size = int(np.ceil(hr_patch_size/scale_factor))

    if not reload_model and lr_patch_size in model_dict:
        model = model_dict[lr_patch_size]
    else:
        model = GraphSuperResolutionNet(crop_size=hr_patch_size, lr_size=lr_patch_size)
        if gpu:
            model = model.cuda()

        path = pathlib.Path(__file__).parent.parent.resolve().__str__() + '/' + model_name
        model = load_model(model, path, gpu=gpu)
        model_dict[lr_patch_size] = model

    padding = 32
    patches_y, patches_x = int(math.ceil(high_res_in.shape[2]/(hr_patch_size - 2 * padding))), int(math.ceil(high_res_in.shape[3]/(hr_patch_size - 2 * padding)))
    out_image = low_res_out.new_zeros((high_res_in.shape[0], high_res_in.shape[1], high_res_in.shape[2], high_res_in.shape[3]))
    bic_lr_image = TT.Resize((high_res_in.shape[2], high_res_in.shape[3]), interpolation=TT.InterpolationMode.BICUBIC)(low_res_out)
    for y in range(patches_y):
        for x in range(patches_x):
            center_patch_size = hr_patch_size - 2 * padding
            hr_start_y, hr_start_x = y * center_patch_size, x * center_patch_size
            patch_hr_start_y, patch_hr_start_x = hr_start_y - padding, hr_start_x - padding

            crop_patch_hr_start_y, crop_patch_hr_start_x = max(patch_hr_start_y, 0), max(patch_hr_start_x, 0)
            crop_patch_hr_end_y, crop_patch_hr_end_x = min(patch_hr_start_y + hr_patch_size, high_res_in.shape[2]), min(patch_hr_start_x + hr_patch_size, high_res_in.shape[3])

            patch_lr_out = bic_lr_image[:, :, crop_patch_hr_start_y:crop_patch_hr_end_y, crop_patch_hr_start_x:crop_patch_hr_end_x]
            patch_hr_in = high_res_in[:, :, crop_patch_hr_start_y:crop_patch_hr_end_y, crop_patch_hr_start_x:crop_patch_hr_end_x]

            start_pad_hr_y, end_pad_hr_y, start_pad_hr_x, end_pad_hr_x = 0, 0, 0, 0
            if patch_hr_in.shape[2] < hr_patch_size or patch_hr_in.shape[3] < hr_patch_size:
                if patch_hr_start_y < 0:
                    start_pad_hr_y = -patch_hr_start_y
                if patch_hr_start_x < 0:
                    start_pad_hr_x = -patch_hr_start_x

                end_pad_hr_y = hr_patch_size - patch_hr_in.shape[2] - start_pad_hr_y
                end_pad_hr_x = hr_patch_size - patch_hr_in.shape[3] - start_pad_hr_x

                patch_lr_out = F.pad(patch_lr_out, (start_pad_hr_x, end_pad_hr_x, start_pad_hr_y, end_pad_hr_y), mode='replicate')
                patch_hr_in = F.pad(patch_hr_in, (start_pad_hr_x, end_pad_hr_x, start_pad_hr_y, end_pad_hr_y), mode='replicate')


            patch_lr_out = TT.Resize((lr_patch_size, lr_patch_size), interpolation=TT.InterpolationMode.BICUBIC)(patch_lr_out)

            patch_out = predict_patch(model, patch_lr_out, patch_hr_in)
            patch_out = patch_out[:, :, padding:-padding, padding:-padding]
            cur_width = min(hr_start_x + center_patch_size, out_image.shape[3]) - hr_start_x
            cur_height = min(hr_start_y + center_patch_size, out_image.shape[2]) - hr_start_y

            out_image[:, :, hr_start_y:(hr_start_y + cur_height), hr_start_x:(hr_start_x + cur_width)] = patch_out[:, :, :cur_height, :cur_width]

    out_image[out_image > 1] = 1
    out_image[out_image < 0] = 0

    return out_image

def save_image(path, img):
    img = img.numpy()
    img[img > 1] = 1
    img[img < 0] = 0
    img = (img * 255).astype(np.uint8)
    imageio.v3.imwrite(path, img)
