"""
TODO
"""
import math


def coord_to_index(lat: float, lon: float, cellsize: float,
                   lat_start: float = 90,
                   lon_start: float = 0) -> tuple[int, int]:
    """
    Converts a point defined by latitude and longitude coordinates
    (must be north and east only) into cell indices of a grid
    of height and width 'cellsize'. By default, grid starts at the
    point 90 degrees N, 0 degrees E corresponding to point (0, 0)
    on the grid. Inverse of index_to_coord function.
    """
    # adjust for grid offset
    lat = lat + (90 - lat_start)
    lon = lon - lon_start

    inv = 1 / cellsize  # scales to grid size
    y = int(90 * inv - math.ceil(lat * inv))
    x = int((360 * inv + math.floor(lon * inv)) % (360 * inv))
    return y, x


def index_to_coord(y: int, x: int, cellsize: float,
                   lat_start: float = 90, lon_start: float = 0,
                   center: bool = False) -> tuple[float, float]:
    """
    Converts index points (y -> latitude, x -> longitude) of a grid
    with height and width of 'cellsize' to the latitude and longitude
    coordinates (in terms of north and east) of either the start
    (top left corner) or middle point within the cell. The point
    90 degrees N, 0 degrees E corresponding to point (0, 0)
    on the grid. Inverse of coord_to_index function.
    """
    offset = 1 / 2 if center else 0
    lat = lat_start - cellsize * (y + offset)
    lon = lon_start + cellsize * (x + offset)
    return lat, lon


def get_neighbors(i: int, j: int,
                  lon_count: int, lat_count: int) -> list[tuple[int, int]]:
    """
    Returns the relative positions of valid neighbor points in a 2D grid.
    """
    pn = [(i - 1, j), (i + 1, j), (i - 1, j - 1), (i, j - 1),
          (i + 1, j - 1), (i - 1, j + 1), (i, j + 1), (i + 1, j + 1)]
    for index, t in enumerate(pn):
        if t[0] < 0 or t[1] < 0 or t[0] >= lon_count or t[1] >= lat_count:
            pn[index] = None
    return [(c[0] - i, c[1] - j) for c in pn if c is not None]
