"""
TODO
"""

def coord_to_index():
    return None

def index_to_coord():
    return None

def get_neighbors(i, j):
    pn = [(i - 1, j), (i + 1, j), (i - 1, j - 1), (i, j - 1),
          (i + 1, j - 1), (i - 1, j + 1), (i, j + 1), (i + 1, j + 1)]
    for index, t in enumerate(pn):
        if t[0] < 0 or t[1] < 0 or t[0] >= self.lon_count or t[1] >= self.lat_count:
            pn[index] = None
    return [(c[0] - i, c[1] - j) for c in pn if c is not None]