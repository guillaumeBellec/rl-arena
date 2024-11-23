import torch
import torch.nn as nn
from typing import List,Union, Any, Optional, cast
import numpy as np
from collections import deque
from dataclasses import dataclass

def to_torch(tensor):
    if isinstance(tensor, list):
        return [to_torch(t) for t in tensor]

    if isinstance(tensor, tuple):
        return tuple([to_torch(t) for t in tensor])

    if isinstance(tensor, dict):
        return dict([(k,to_torch(v)) for k, v in tensor.items()])

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    return tensor

def make_vgg_layers(cfg: List[Union[str, int]], in_channels=3, batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def to_scalar(val):
    if isinstance(val, torch.Tensor):
        val = float(torch.mean(val).item())

    if isinstance(val, np.ndarray):
        return float(np.mean(val))

    if np.isscalar(val):
        return val

    raise NotImplementedError()

class EmaVal:

    def __init__(self, decay=0.99):
        self._val = None
        self.d = decay

    def read(self):
        if self._val is None: return -1.0
        return self._val


    def __call__(self, val):
        if isinstance(val, torch.Tensor):
            val = float(torch.mean(val).item())

        if self._val is not None:
            assert np.isscalar(self._val), f"unexpected type: {type(self._val)}: {self._val}"

        if self._val is None: self._val = val # no value stored, overwritten memory
        self._val = self.d * self._val + (1-self.d) * float(val)
        return self._val


def deepclone(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.clone()

    if isinstance(tensor, list):
        return [deepclone(t) for t in tensor]

    if isinstance(tensor, tuple):
        return tuple([deepclone(t) for t in tensor])

    if isinstance(tensor, dict):
        results = {}
        for k,v in tensor.items():
            results[k] = deepclone(v)
        return results

    if isinstance(tensor, np.ndarray):
        return tensor

    if np.isscalar(tensor):
        return tensor

    if tensor is None:
        return None

    raise NotImplemented()


def deepdetach(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()

    if isinstance(tensor, list):
        return [deepclone(t) for t in tensor]

    if isinstance(tensor, tuple):
        return tuple([deepclone(t) for t in tensor])

    if isinstance(tensor, dict):
        results = {}
        for k,v in tensor.items():
            results[k] = deepclone(v)
        return results

    if isinstance(tensor, np.ndarray):
        return tensor

    if np.isscalar(tensor):
        return tensor

    if tensor is None:
        return tensor

    raise NotImplemented()