from math import log

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .functional import create_fixed_cupy_sparse_matrices, GraphQuadraticSolver

#INPUT_DIM = 2
INPUT_DIM = 1
FEATURE_DIM = 64


def distance_norm(left, right, norm):
    return torch.mean(torch.abs(left - right) ** norm, dim=1, keepdim=True)


def get_neighbor_affinity_no_border(feature_map, mu, lambda_):
    B, M, H, W = feature_map.shape

    feature_map_padded = F.pad(feature_map, (1, 1, 1, 1), 'constant', 0)

    top = distance_norm(feature_map_padded[:, :, 0:-2, 1:-1], feature_map, norm=1.5)
    bottom = distance_norm(feature_map_padded[:, :, 2:, 1:-1], feature_map, norm=1.5)
    left = distance_norm(feature_map_padded[:, :, 1:-1, 0:-2], feature_map, norm=1.5)
    right = distance_norm(feature_map_padded[:, :, 1:-1, 2:], feature_map, norm=1.5)

    affinity = torch.cat([top, bottom, left, right], dim=1) / (1e-6 + mu ** 2)
    affinity = torch.exp(-affinity)

    border_remover = torch.ones((1, 4, H, W), device=feature_map.device)
    border_remover[0, 0, 0, :] = 0  # top
    border_remover[0, 1, -1, :] = 0  # bottom
    border_remover[0, 2, :, 0] = 0  # left
    border_remover[0, 3, :, -1] = 0  # right

    affinity = affinity * border_remover
    center = torch.sum(affinity, dim=1, keepdim=True)
    affinity = torch.cat([affinity, center], dim=1)
    affinity = affinity * lambda_

    return affinity


class GraphSuperResolutionNet(nn.Module):

    def __init__(
            self,
            crop_size=256,
            lr_size=32,
            feature_extractor='UResNet',
            pretrained=True,
            lambda_init=1.0,
            mu_init=0.1
    ):
        super().__init__()

        if crop_size not in [64, 128, 256, 512]:
            raise ValueError('Crop size should be in {64, 128, 256, 512}, got ' + str(crop_size))

        if feature_extractor == 'Color':
            self.feature_extractor = None
            # so the optimizer does not complain in case we have no other parameters
            self.dummy_param = nn.Parameter(torch.zeros(1))
        elif feature_extractor == 'UResNet':
            self.feature_extractor = smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=INPUT_DIM,
                                              encoder_weights='imagenet' if pretrained else None)
        elif feature_extractor == 'UResNet18':
            self.feature_extractor = smp.Unet('resnet18', classes=FEATURE_DIM, in_channels=INPUT_DIM,
                                              encoder_weights='imagenet' if pretrained else None)
        elif feature_extractor == 'UEffNet2':
            self.feature_extractor = smp.Unet('efficientnet-b2', classes=FEATURE_DIM, in_channels=INPUT_DIM,
                                              encoder_weights='imagenet' if pretrained else None)
        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')

        self.log_lambda = nn.Parameter(torch.tensor([log(lambda_init)]))
        self.log_mu = nn.Parameter(torch.tensor([log(mu_init)]))
        self.mx_dict = create_fixed_cupy_sparse_matrices(crop_size, crop_size, lr_size, lr_size)

    def forward(self, guide, source):

        if self.feature_extractor is None:
            pixel_features = guide
        else:
            #pixel_features = self.feature_extractor(torch.cat([guide, upsampled_source], dim=1))
            pixel_features = self.feature_extractor(guide)

        torch.cuda.synchronize()

        mu, lambda_ = torch.exp(self.log_mu), torch.exp(self.log_lambda)
        neighbor_affinity = get_neighbor_affinity_no_border(pixel_features, mu, lambda_)

        # Optimizing takes basically all of the time -> 800ms vs 10ms
        #start.record()
        y_pred = GraphQuadraticSolver.apply(neighbor_affinity, source, self.mx_dict)
        #end.record()
        #torch.cuda.synchronize()
        #print("Opt", start.elapsed_time(end))

        return {'y_pred': y_pred, 'neighbor_affinity': neighbor_affinity}