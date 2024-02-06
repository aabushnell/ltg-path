import os
from itertools import product

import numpy as np

import grid_builder as gb
import grid_helpers as gh
import file_reader as fr
import rasterize_rivers as rr

CELL_SIZE = 1 / 12
INV_CELL = 12
COARSE_DIM = 24
OUTPUT_FOLDER = '/Users/aaron/Documents/Work/Data/RAW_COST_DATA5/'

elev_full = gb.DataGrid(fr.read_elevation_nc(), CELL_SIZE, 90, 0)
landlake_full = gb.DataGrid(fr.read_landlake_nc(), CELL_SIZE, 90, 0)
river_full = gb.DataGrid(fr.read_rivers_nc(), CELL_SIZE, 90, 0)

land_cells = []
coast_cells = []
sea_cells = []

for y_coarse, x_coarse in product(range(9, 45),
                                  range(90)):
    if x_coarse < 13:
        x_coarse += 167
    else:
        x_coarse -= 13

    y_start = y_coarse * COARSE_DIM
    x_start = x_coarse * COARSE_DIM

    y_end = y_start + (COARSE_DIM - 1)
    x_end = x_start + (COARSE_DIM - 1)

    grid_id = y_coarse * 180 + x_coarse

    landlake = landlake_full.subgrid(y_start, x_start, y_end, x_end)

    if (landlake.as_matrix() > 0).any():
        if (landlake.as_matrix() < 0).any():
            coast_cells.append(grid_id)
        else:
            land_cells.append(grid_id)
    else:
        sea_cells.append(grid_id)

land_cells = np.array(land_cells)
land_cells.tofile('land.csv', sep=',')
coast_cells = np.array(coast_cells)
coast_cells.tofile('coast.csv', sep=',')
sea_cells = np.array(sea_cells)
sea_cells.tofile('sea.csv', sep=',')
