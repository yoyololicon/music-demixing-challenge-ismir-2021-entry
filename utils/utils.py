from torch import nn


def remove_weight_norms(m):
    if hasattr(m, 'weight_g'):
        nn.utils.remove_weight_norm(m)


def add_weight_norms(m):
    if hasattr(m, 'weight'):
        nn.utils.weight_norm(m)


def get_instance(module, config, *args, **kwargs):
    return getattr(module, config['type'])(*args, **config['args'], **kwargs)
