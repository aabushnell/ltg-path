"""
TODO
"""
import numpy as np

coeffs = {
    'flat': 1,  # cost per km (horizontal)
    'up': 1/120,  # cost per m (vertical)
    'down_mod': 1/360,  # cost per m (vertical)
    'down_steep_cutoff': -0.2125,  # angle
    'down_steep': -1/360,  # cost per m (vertical)
    'coastal_sea': 0.9,  # cost per km (horizontal) premium
    'deep_sea': 0.5,
    'loading': 3.6,  # == 1hr at 1m/s
    'river': 0.9,  # cost per km (horizontal) premium
}

ELEVATION_MULTIPLIER = 10


def transport_method_cost(distance: float,
                          elevation_start: int,
                          elevation_end: int,
                          terrain_start: int,
                          terrain_end: int,
                          river_travel: bool,
                          deep_sea_val: int = -9999) -> float:
    """
    Calculates the total 'cost' of moving along a given distance
    modified by predefined parameters and taking into account
    elevation change and implicit movement from land to sea
    (where any negative elevation is considered to be sea).
    Also features a special cost for travel between adjacent
    river nodes.
    """
    if (terrain_start == deep_sea_val
            or terrain_end == deep_sea_val):  # deep sea travel
        cost = 1e30
    elif terrain_start == 1 and terrain_end == 1:  # land travel
        cost = (coeffs['flat'] * distance
                + (elevation_change_cost(elevation_start, elevation_end)))
        if river_travel:
            # decreased cost for travel along major rivers
            cost *= coeffs['river']
    elif terrain_start <= 0 and terrain_end <= 0:  # shallow water travel
        cost = coeffs['coastal_sea'] * coeffs['flat'] * distance
    else:  # loading/unloading (from land to sea or sea to land)
        cost = coeffs['loading'] + coeffs['flat'] * distance
    return cost


def elevation_change_cost(elevation_start: int,
                          elevation_end: int) -> float:
    """
    Returns the 'cost premium' for travel with changing elevations
    using predifined parameters.
    :param elevation_start: starting elevation (in meters)
    :param elevation_end: ending elevation (in meters)
    """
    delta = elevation_end - elevation_start
    if delta == 0:
        return 0
    elif delta > 0:
        return coeffs['up'] * delta * ELEVATION_MULTIPLIER
    elif 0 > delta >= coeffs['down_steep_cutoff']:
        return coeffs['down_mod'] * delta * ELEVATION_MULTIPLIER
    else:
        return coeffs['down_steep'] * delta * ELEVATION_MULTIPLIER


def high_seas_cost(distance: float,
                   terrain_start: int,
                   terrain_end: int) -> float:
    if terrain_start == 1 or terrain_end == 1:  # land travel
        cost = 1e30
    else:
        cost = coeffs['deep_sea'] * coeffs['flat'] * distance
    return cost
