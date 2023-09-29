"""
TODO
"""

coeffs = {
    'flat': 0.72,  # meters/second
    'up': 6,  # meters/hour
    'down_mod': 2,  # meters/hour
    'down_steep_cutoff': -0.2125,  # angle
    'down_steep': -2,  # meters/hour
    'sea': 0.9,  # meters/sec
    'loading': 1,  # equivalent to 1 hr of transport
    'river': 0.9,  # meters/sec
}

UNIT_CONV = 1000 / 3600  # km/h to m/s


def transport_method_cost(distance: float,
                          elevation_start: int,
                          elevation_end: int,
                          river_travel: bool) -> float:
    """
    Calculates the total 'cost' of moving along a given distance
    modified by predefined parameters and taking into account
    elevation change and implicit movement from land to sea
    (where any negative elevation is considered to be sea).
    Also features a special cost for travel between adjacent
    river nodes.
    """
    if elevation_start >= 0 and elevation_end >= 0:
        if river_travel:
            cost = coeffs['river'] * coeffs['flat'] * distance * UNIT_CONV
        else:
            cost = (coeffs['flat'] * distance * UNIT_CONV
                    + elevation_change_cost(elevation_start, elevation_end))
    elif elevation_start < 0 and elevation_end < 0:
        cost = coeffs['sea'] * coeffs['flat'] * distance * UNIT_CONV
    else:
        cost = coeffs['loading'] + coeffs['flat'] * distance * UNIT_CONV
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
