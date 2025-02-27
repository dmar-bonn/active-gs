from .random import Random
from .exploration import Exploration
from .confidence import Confidence


def get_planner(cfg, device):
    planner_cfg = cfg.planner
    if planner_cfg.type == "random":
        return Random(planner_cfg, device)
    elif planner_cfg.type == "exploration":
        return Exploration(planner_cfg, device)
    elif planner_cfg.type == "confidence":
        return Confidence(planner_cfg, device)
    else:
        raise NotImplementedError
