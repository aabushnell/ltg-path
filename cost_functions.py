"""
TODO
"""

coeffs = {
    "flat" : 0.72,
    "up": 6,
    "down_mod": 2,
    "down_steep_cutoff": -0.2125,
    "down_steep": -2,
    "sea": 0.9,
    "loading": 3.6,
}

def transport_method_cost(distance, elevation_start, elevation_end):
    if elevation_start >= 0 and elevation_end >= 0:
        cost = coeffs['flat'] * distance + elevation_change_cost(elevation_start, elevation_end)
    elif elevation_start < 0 and elevation_end < 0:
        cost = coeffs['sea'] * coeffs['flat'] * distance
    else:
        cost = coeffs['loading'] + coeffs['flat'] * distance
    return cost

def elevation_change_cost(elevation_start, elevation_end):
    delta = elevation_end - elevation_start
    if delta == 0: return 0
    elif delta > 0: return coeffs['up'] * delta
    elif 0 > delta >= coeffs['down_steep_cutoff']: return coeffs['down_mod'] * delta
    else: return coeffs['down_steep'] * delta
