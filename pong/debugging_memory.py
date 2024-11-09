import sys
import gc
import torch
import numpy as np
from collections import defaultdict
import psutil
import os


def get_tensor_size(tensor):
    """Calculate memory usage of a PyTorch tensor including gradients."""
    size = tensor.element_size() * tensor.nelement()

    # Add gradient memory if exists
    if tensor.grad is not None:
        size += tensor.grad.element_size() * tensor.grad.nelement()

    return size


def get_object_size(obj, seen=None):
    """Recursively calculate size of object and its children in bytes."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if torch.is_tensor(obj):
        return get_tensor_size(obj)

    if isinstance(obj, (list, tuple, set, dict)):
        size += sum(get_object_size(item, seen) for item in obj)

    return size


def memory_profile():
    """Profile current PyTorch memory usage."""
    memory_stats = defaultdict(int)

    # Get all objects tracked by the garbage collector
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            memory_stats['tensor_count'] += 1
            memory_stats['total_tensor_memory'] += get_tensor_size(obj)

            # Track tensors with gradients
            if obj.grad is not None:
                memory_stats['tensors_with_grad'] += 1
                memory_stats['grad_memory'] += obj.grad.element_size() * obj.grad.nelement()

    # Get process memory info
    process = psutil.Process(os.getpid())
    memory_stats['process_memory'] = process.memory_info().rss

    # Get PyTorch allocated memory
    if torch.cuda.is_available():
        memory_stats['cuda_allocated'] = torch.cuda.memory_allocated()
        memory_stats['cuda_cached'] = torch.cuda.memory_reserved()

    return memory_stats


def find_memory_leaks(iterations=5, func=None):
    """Monitor memory usage over iterations to detect leaks."""
    initial_memory = memory_profile()
    memory_changes = []

    for i in range(iterations):
        if func:
            func()

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        current_memory = memory_profile()
        memory_changes.append({
            key: current_memory[key] - initial_memory[key]
            for key in current_memory
        })

    return memory_changes