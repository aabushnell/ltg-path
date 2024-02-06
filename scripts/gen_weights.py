from itertools import product

import numpy as np

import grid_builder as gb
import grid_helpers as gh
import file_reader as fr

CELL_SIZE = 1 / 12
COARSE_DIM = 24
OUTPUT_FILE = ('/Users/aaron/Documents/Work/Code/Pathfinding/data/'
               + 'weights_sea.nc')

landlake_full = gb.DataGrid(fr.read_landlake_nc(), CELL_SIZE, 90, 0)

weights = np.zeros((2160, 4320))

for y_coarse, x_coarse in product(range(0, 90), range(0, 180)):
    y_start = y_coarse * COARSE_DIM
    x_start = x_coarse * COARSE_DIM

    y_end = y_start + (COARSE_DIM - 1)
    x_end = x_start + (COARSE_DIM - 1)

    grid_id = y_coarse * 180 + x_coarse

    landlake = landlake_full.subgrid(y_start, x_start, y_end, x_end)

    land_count = np.count_nonzero(landlake.as_matrix() > -1)
    if land_count < 576:
        per_cell_weight = 1 / (576 - land_count)
    else:
        per_cell_weight = 0

    for y_fine, x_fine in product(range(0, 24), range(0, 24)):
        if landlake.loc(y_fine, x_fine) < 0:
            weight = per_cell_weight
        else:
            weight = 0

        weights[y_start + y_fine, x_start + x_fine] = weight

fr.save_nc(OUTPUT_FILE, weights)
