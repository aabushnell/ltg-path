"""
TODO
"""
from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from grid_builder import DataGrid


def coord_to_index(lat: float, lon: float, cell_size: float,
                   lat_start: float = 90,
                   lon_start: float = 0) -> tuple[int, int]:
    """
    Converts a point defined by latitude and longitude coordinates
    (must be north and east only) into cell indices of a grid
    of height and width 'cell_size'. By default, grid starts at the
    point 90 degrees N, 0 degrees E corresponding to point (0, 0)
    on the grid. Inverse of index_to_coord function.
    """
    # adjust for grid offset
    lat = lat + (90 - lat_start)
    lon = lon - lon_start

    inv = 1 / cell_size  # scales to grid size
    y = int(90 * inv - math.ceil(lat * inv))
    x = int((360 * inv + math.floor(lon * inv)) % (360 * inv))
    return y, x


def index_to_coord(y: int, x: int, cell_size: float,
                   lat_start: float = 90, lon_start: float = 0,
                   center: bool = False,
                   normalize_lon: bool = False) -> tuple[float, float]:
    """
    Converts index points (y -> latitude, x -> longitude) of a grid
    with height and width of 'cell_size' to the latitude and longitude
    coordinates (in terms of north and east) of either the start
    (top left corner) or middle point within the cell. The point
    90 degrees N, 0 degrees E corresponding to point (0, 0)
    on the grid. Inverse of coord_to_index function.
    """
    offset = 1 / 2 if center else 0
    lat = lat_start - cell_size * (y + offset)
    lon = (lon_start + cell_size * (x + offset)) % 360
    if normalize_lon and lon > 180:
        lon -= 360
    return lat, lon


def get_valid_neighbors(y_pos: int,
                        x_pos: int,
                        depth: int,
                        y_dim: int,
                        x_dim: int,
                        overflow: bool = False) -> list[tuple[int, int]]:
    """
    Returns the relative positions of valid neighbor points in a 2D grid.
    :param overflow: Allows neighboring points to roll over past zero
    :param y_pos: The y-coordinate of center point in grid
    :param x_pos: The x-coordinate of center point in grid
    :param depth: The number of steps away from center point to search
    :param y_dim: The number of points in grid along y-axis
    :param x_dim: The number of points in grid along x-axis
    :return: A list of tuples (y-position, x-position) of valid neighbors
    """
    if overflow:
        return [((y_pos + dy + y_dim) % y_dim,
                 (x_pos + dx + x_dim) % x_dim) for dy, dx
                in product(range(-1 * depth, depth + 1),
                           range(-1 * depth, depth + 1))
                if (dy != 0 or dx != 0)]
    else:
        return [(y_pos + dy, x_pos + dx) for dy, dx
                in product(range(-1 * depth, depth + 1),
                           range(-1 * depth, depth + 1))
                if (dy != 0 or dx != 0)
                and 0 <= y_pos + dy < y_dim
                and 0 <= x_pos + dx < x_dim]


def get_valid_neigbors_split(y_pos: int, x_pos: int, depth: int,
                             y_dim: int, x_dim: int
                             ) -> tuple[list[int], list[int]]:
    """
    Returns the relative positions of valid neighbor points in a 2D grid
    formatted as a tuple of two lists with corresponding y and x
    coordinates of points.
    :param y_pos: The y-coordinate of center point in grid
    :param x_pos: The x-coordinate of center point in grid
    :param depth: The number of steps away from center point to search
    :param y_dim: The number of points in grid along y-axis
    :param x_dim: The number of points in grid along x-axis
    :return: A tuple of lists of y and x coordinates (respectively)
    that together define the points of valid neighbors.
    """
    pn = get_valid_neighbors(y_pos, x_pos, depth, y_dim, x_dim)

    ys = [x[0] for x in pn]
    xs = [x[1] for x in pn]

    return ys, xs


def get_offset_subarray(grid: np.ndarray[np.ndarray[int]],
                        y_offset: int,
                        x_offset: int) -> np.ndarray[np.ndarray[int]]:
    if grid.shape[0] != grid.shape[1]:
        print('Grid dimensions must be symmetric (NxN)')
        raise ValueError
    else:
        grid_dim = int(grid.shape[0] / 3)
    row_start = grid_dim * (1 + y_offset)
    row_end = grid_dim * (2 + y_offset)
    col_start = grid_dim * (1 + x_offset)
    col_end = grid_dim * (2 + x_offset)

    return grid[row_start:row_end, col_start:col_end]


def all_outside_subarrays(grid: np.ndarray[np.ndarray[int]]
                          ) -> list[np.ndarray[np.ndarray[int]]]:
    neighbors = [(i, j) for i in range(-1, 2) for j in range(-1, 2)
                 if i != 0 or j != 0]

    array_list = []
    for y_off, x_off in neighbors:
        array_list.append(get_offset_subarray(grid, y_off, x_off))
    return array_list


def mask_array(supergrid: np.ndarray[np.ndarray[int]],
               mask: np.ndarray[np.ndarray[int]],
               values: np.ndarray[np.ndarray[int]]
               ) -> np.ndarray[np.ndarray[int]]:
    if supergrid.shape != values.shape:
        raise ValueError

    ul = np.where(supergrid == mask[0, 0])
    y1 = ul[0][0]
    x1 = ul[1][0]
    lr = np.where(supergrid == mask[-1, -1])
    y2 = lr[0][0]
    x2 = lr[1][0]

    return values[y1:(y2 + 1), x1:(x2 + 1)]


def combine_grid_neighbors(top_left: DataGrid,
                           top: DataGrid,
                           top_right: DataGrid,
                           left: DataGrid,
                           center: DataGrid,
                           right: DataGrid,
                           bottom_left: DataGrid,
                           bottom: DataGrid,
                           bottom_right: DataGrid) -> DataGrid:
    return ((top_left
             .join(top, side='r')
             .join(top_right, side='r'))
            .join((left
                   .join(center, side='r')
                   .join(right, side='r')), side='b')
            .join((bottom_left
                   .join(bottom, side='r')
                   .join(bottom_right, side='r')), side='b'))


def get_grid_neighbors(data: DataGrid,
                       y_start: int, x_start: int,
                       y_end: int, x_end: int,
                       coarse_dim: int) -> DataGrid:
    offsets = [(y, x) for (y, x) in product(range(-1, 2), range(-1, 2))]
    subgrids = [data.subgrid(y_start + (y * coarse_dim),
                             (x_start + (x * coarse_dim)
                              + 4320) % 4320,
                             y_end + (y * coarse_dim),
                             (x_end + (x * coarse_dim)
                              + 4320) % 4320)
                for (y, x) in offsets]
    return combine_grid_neighbors(*subgrids)
