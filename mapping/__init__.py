from .mapper import IncrementalMapper


def get_mapper(cfg, device):
    return IncrementalMapper(cfg.mapper, device)
