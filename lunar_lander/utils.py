import torch
import torch.nn as nn
from typing import List,Union, Any, Optional, cast
import numpy as np
from collections import deque
from dataclasses import dataclass, is_dataclass, fields

from typing import Any, Dict, List, Tuple, Union, Callable


def make_deep(fn: Callable) -> Callable:
    """
    Creates a deep version of a function that works on single tensor inputs.
    The function will be applied recursively to all tensors in nested structures.

    Args:
        fn: Function that takes a single tensor as input

    Returns:
        Deep version of the function that works on nested structures
    """

    def deep_fn(struct: Any, *args, **kwargs) -> Any:
        if isinstance(struct, torch.Tensor):
            return fn(struct, *args, **kwargs)
        elif isinstance(struct, np.ndarray):
            return fn(struct, *args, **kwargs)
        elif isinstance(struct, dict):
            return {k: deep_fn(v, *args, **kwargs) for k, v in struct.items()}
        elif isinstance(struct, list):
            return [deep_fn(x, *args, **kwargs) for x in struct]
        elif isinstance(struct, deque):
            Q = deque(maxlen=struct.maxlen)
            for e in struct: Q.append(deep_fn(e, *args, **kwargs))
            return Q
        elif isinstance(struct, tuple):
            # Handle named tuples
            if hasattr(struct, '_fields'):
                return type(struct)(*[deep_fn(x, *args, **kwargs) for x in struct])
            return tuple(deep_fn(x, *args, **kwargs) for x in struct)
        elif is_dataclass(struct):
            # Handle dataclasses by recreating them with processed fields
            return type(struct)(**{
                field.name: deep_fn(getattr(struct, field.name), **kwargs)
                for field in fields(struct)
            })
        elif struct is None:
            return None # just return None
        elif np.isscalar(struct):
            # if None or float or int?
            return fn(struct, *args, **kwargs)
        else:
            raise NotImplementedError()

    return deep_fn

def make_deep_for_lists(fn: Callable) -> Callable:
    """
    Creates a deep version of a function that works on lists of tensors.
    The function will be applied recursively to all lists of tensors in nested structures.

    Args:
        fn: Function that takes a list of tensors as input (like torch.stack or torch.cat)

    Returns:
        Deep version of the function that works on nested structures
    """

    def deep_fn(batch: List[Any], *args, **kwargs) -> Any:
        if not batch:
            return batch

        sample = batch[0]

        if isinstance(sample, torch.Tensor):
            return fn(batch, *args, **kwargs)
        elif isinstance(sample, np.ndarray):
            return fn(batch, *args, **kwargs)
        elif isinstance(sample, dict):
            return {k: deep_fn([b[k] for b in batch], *args, **kwargs) for k in sample}
        elif isinstance(sample, list):
            return [deep_fn([b[i] for b in batch], *args, **kwargs) for i in range(len(sample))]

        elif isinstance(sample, deque):
            Q = deque(maxlen=sample.maxlen)
            for i in range(len(sample)):
                Q.append(deep_fn([b[i] for b in batch], *args, **kwargs))
            return Q

        elif isinstance(sample, tuple):
            if hasattr(sample, '_fields'):  # Named tuple
                return type(sample)(*[
                    deep_fn([b[i] for b in batch], *args, **kwargs)
                    for i in range(len(sample))
                ])
            return tuple(deep_fn([b[i] for b in batch],*args,  **kwargs) for i in range(len(sample)))
        elif is_dataclass(sample):
            # Handle dataclasses by processing each field separately
            return type(sample)(**{
                field.name: deep_fn([getattr(b, field.name) for b in batch], **kwargs)
                for field in fields(sample)
            })
        elif sample is None:
            for b in batch: assert b is None
            return None # just return None
        elif np.isscalar(sample):
            for b in batch: assert b == sample # should be all the same?
            # float or int?
            return sample #deep_fn([np.array(e) for e in batch], *args, **kwargs)
        else:
            raise NotImplementedError()

    return deep_fn

def _to_torch_single_element(tensor):

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    return tensor

to_torch = make_deep(_to_torch_single_element)


def _to_numpy_single_element(tensor):

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor

to_numpy = make_deep(_to_numpy_single_element)


deepstack = make_deep_for_lists(torch.stack)
deepconcatenate = make_deep_for_lists(torch.concatenate)


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

deepclone = make_deep(torch.clone)

deepdetach = make_deep(torch.detach)

deeppermute = make_deep(torch.permute)

@DeprecationWarning
class ReplayBuffer:

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

    @staticmethod
    def merge_tensor_stacks(simulations, keep_i, n):
        """
        Get a list of Simulation objects, it will generate stack of torch tensors concatenated on the batch dimension.


        :param simulations:
        :param keep_i:
        :param n:
        :return:
        """
        # sub-select those with good size
        simulations = [simulations[keep_i]]
        js = np.random.permutation(len(simulations)-1)[:n]
        for j in js:
            index = (keep_i + 1 + j) % len(simulations)
            assert index != keep_i
            assert 0 <= index < len(simulations)
            simulations.append(simulations[index])

        tensor_stacks = [simu.replay_buffer.get_stack_tensors() for simu in simulations]
        tensor_lists = [[] for tensor in tensor_stacks[0]]

        for tensor_stack in tensor_stacks:
            for i in range(len(tensor_lists)):
                tensor_lists[i].append(tensor_stack[i])

        agent_states = None
        if simulations[0].agent_states is not None:
            agent_states = [simu.agent_states for simu in simulations]
            assert len(agent_states[0]) == 2, "only implemented for LSTM states"
            hs = torch.cat([simu.agent_states[0] for simu in simulations], 1) # RNN states have batch on second dim
            cs = torch.cat([simu.agent_states[1] for simu in simulations], 1)
            agent_states = (hs, cs)

        return [torch.cat(l,0) for l in tensor_lists], agent_states

    def is_empty(self):
        return len(self._deque) == 0

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


def _single_shape(tensor):
    return tensor.shape

deepshape = make_deep(_single_shape)

# run as main
if __name__ == "__main__":

    make_data = lambda i : torch.arange(i,i+10)

    examples = [{
        "my_key": (make_data(i), make_data(i))
    }
    for i in range(4)
    ]

    results = deepstack(examples)
    print(results)
