from .habitat_simulator import HabitatSimulator


def get_simulator(cfg):
    return HabitatSimulator(cfg.simulator, cfg.scene)
