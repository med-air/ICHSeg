# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

from typing import Tuple, Union, List

from batchgenerators.transforms.abstract_transforms import AbstractTransform


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self, apply_to_keys: Union[List[str], Tuple[str]] = ('data', 'seg')):
        """
        Transforms a 5D array (b, c, x, y, z) to a 4D array (b, c * x, y, z) by overloading the color channel
        """
        self.apply_to_keys = apply_to_keys

    def __call__(self, **data_dict):
        for k in self.apply_to_keys:
            shp = data_dict[k].shape
            assert len(shp) == 5, 'This transform only works on 3D data, so expects 5D tensor (b, c, x, y, z) as input.'
            data_dict[k] = data_dict[k].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
            shape_key = f'orig_shape_{k}'
            assert shape_key not in data_dict.keys(), f'Convert3DTo2DTransform needs to store the original shape. ' \
                                                      f'It does that using the {shape_key} key. That key is ' \
                                                      f'already taken. Bummer.'
            data_dict[shape_key] = shp
        return data_dict


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self, apply_to_keys: Union[List[str], Tuple[str]] = ('data', 'seg')):
        """
        Reverts Convert3DTo2DTransform by transforming a 4D array (b, c * x, y, z) back to 5D  (b, c, x, y, z)
        """
        self.apply_to_keys = apply_to_keys

    def __call__(self, **data_dict):
        for k in self.apply_to_keys:
            shape_key = f'orig_shape_{k}'
            assert shape_key in data_dict.keys(), f'Did not find key {shape_key} in data_dict. Shitty. ' \
                                                  f'Convert2DTo3DTransform only works in tandem with ' \
                                                  f'Convert3DTo2DTransform and you probably forgot to add ' \
                                                  f'Convert3DTo2DTransform to your pipeline. (Convert3DTo2DTransform ' \
                                                  f'is where the missing key is generated)'
            original_shape = data_dict[shape_key]
            current_shape = data_dict[k].shape
            data_dict[k] = data_dict[k].reshape((original_shape[0], original_shape[1], original_shape[2],
                                                 current_shape[-2], current_shape[-1]))
        return data_dict