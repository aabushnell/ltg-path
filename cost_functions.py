"""
TODO
"""

coeffs = {
    'flat': 1,  # cost per meter (horizontal)
    'up': 25/3,  # cost per meter (vertical)
    'down_mod': 25/9,  # cost per meter (vertical)
    'down_steep_cutoff': -0.2125,  # angle
    'down_steep': -25/9,  # cost per meter (vertical)
    'sea': 0.9,  # cost per meter (horizontal) premium
    'loading': 3600,  # equivalent to 1 hr of transport
    'river': 0.9,  # cost per meter (horizontal) premium
}

KM_TO_M = 1000  # kilometers to meters conversion


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
    if elevation_start >= 0 and elevation_end >= 0:  # land travel
        cost = (coeffs['flat'] * distance * KM_TO_M
                + elevation_change_cost(elevation_start, elevation_end))
        if river_travel:
            # decreased cost for travel along major rivers
            cost *= coeffs['river']
    elif elevation_start < 0 and elevation_end < 0:  # ocean travel
        cost = coeffs['sea'] * coeffs['flat'] * distance * KM_TO_M
    else:  # loading/unloading (from land to sea or sea to land)
        cost = coeffs['loading'] + coeffs['flat'] * distance * KM_TO_M
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
