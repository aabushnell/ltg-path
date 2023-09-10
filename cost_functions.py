"""
TODO
"""

coeffs = {
    'flat': 0.72,
    'up': 6,
    'down_mod': 2,
    'down_steep_cutoff': -0.2125,
    'down_steep': -2,
    'sea': 0.9,
    'loading': 3.6,
    }


def transport_method_cost(distance: float,
                          elevation_start: int,
                          elevation_end: int) -> float:
    """
    Calculates the total 'cost' of moving along a given distance
    modified by predefined parameters and taking into account
    elevation change and implicit movement from land to sea
    (where any negative elevation is considered to be sea).
    """
    if elevation_start >= 0 and elevation_end >= 0:
        cost = coeffs['flat'] * distance + elevation_change_cost(
            elevation_start, elevation_end)
    elif elevation_start < 0 and elevation_end < 0:
        cost = coeffs['sea'] * coeffs['flat'] * distance
    else:
        cost = coeffs['loading'] + coeffs['flat'] * distance
    return cost


def elevation_change_cost(elevation_start: int,
                          elevation_end: int) -> float:
    """
    Returns the 'cost premium' for travel with changing elevations
    using predifined parameters.
    """
    delta = elevation_end - elevation_start
    if delta == 0:
        return 0
    elif delta > 0:
        return coeffs['up'] * delta
    elif 0 > delta >= coeffs['down_steep_cutoff']:
        return coeffs['down_mod'] * delta
    else:
        return coeffs['down_steep'] * delta
