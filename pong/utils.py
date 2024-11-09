import torch
import torch.nn as nn
from typing import List,Union, Any, Optional, cast
import numpy as np
from collections import deque
from dataclasses import dataclass

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

    raise NotImplemented()



class RunStack:

    def __init__(self, max_len):
        self._deque = deque(maxlen=max_len)

    def add(self, runs):
        self._deque.append(deepdetach(runs))

    def convert_to_torch(self, tensor):
        if not isinstance(tensor, np.ndarray): raise NotImplementedError()

        tensor = torch.from_numpy(tensor)
        if tensor.is_inference():
            tensor = tensor.clone()

        return tensor

    def get_stack_tensors(self):
        runs = list(self._deque)
        first_run = runs[0]
        assert isinstance(first_run, tuple)

        data_lists = [[] for _ in range(len(first_run))]
        for i in range(len(first_run)):
            for run in runs:
                data_lists[i].append(run[i])

        data_tensors = [np.concatenate(l, 1) for l in data_lists]

        data_torch_tensors = [self.convert_to_torch(arr) for arr in data_tensors]

        return tuple(data_torch_tensors)

@dataclass
class Simulation:
    envs: any
    agent_states: any
    last_observations: any
    last_dones : any
    run_stack: any