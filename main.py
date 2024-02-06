import os
from itertools import product

import numpy as np

import grid_builder as gb
import grid_helpers as gh
import file_reader as fr
import rasterize_rivers as rr

CELL_SIZE = 1 / 12
COARSE_DIM = 24
OUTPUT_FOLDER = '/Users/aaron/Documents/Work/Data/RAW_COST_DATA_SEA/'

elev_full = gb.DataGrid(fr.read_elevation_nc(), CELL_SIZE, 90, 0)
landlake_full = gb.DataGrid(fr.read_landlake_nc(), CELL_SIZE, 90, 0)
river_full = gb.DataGrid(fr.read_rivers_nc(), CELL_SIZE, 90, 0)

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

    grid_id = y_coarse * 1000 + x_coarse

    elev_combined = gh.get_grid_neighbors(elev_full, y_start, x_start,
                                          y_end, x_end, COARSE_DIM)
    landlake_combined = gh.get_grid_neighbors(landlake_full, y_start, x_start,
                                              y_end, x_end, COARSE_DIM)
    river_combined = gh.get_grid_neighbors(river_full, y_start, x_start,
                                           y_end, x_end, COARSE_DIM)

    lat_start, lon_start = gh.index_to_coord(y_start - COARSE_DIM,
                                             x_start - COARSE_DIM,
                                             CELL_SIZE)

    combined_grid = gb.BilateralCostGridSea(grid_id,
                                         CELL_SIZE,
                                         lat_start,
                                         lon_start,
                                         elev_combined,
                                         landlake_combined,
                                         river_combined,
                                         deep_sea_depth=3)

    combined_grid.build_cost_matrix()
    output = combined_grid.bilateral_matrix_output(flat_inner_matrix=True)

    if not os.path.exists(OUTPUT_FOLDER + f'{y_coarse}'):
        os.mkdir(OUTPUT_FOLDER + f'{y_coarse}')
    output.to_file(OUTPUT_FOLDER + f'{y_coarse}/{x_coarse}.nc')

    print(f'grid ({x_coarse}, {y_coarse}) processed')

# Y_START = 626
# X_START = 1466

# Y_START = 480
# X_START = 20

# print(gh.index_to_coord(Y_START, X_START, CELL_SIZE))
#
# y_end = Y_START + COARSE_DIM * 3 - 1
# x_end = X_START + COARSE_DIM * 3 - 1
#
# grid = full_elev_grid.subgrid(1, Y_START, X_START, y_end, x_end)
#
# print(grid.grid)
# print(grid.elev_arr)
# print(grid.landlake_arr)
# grid.mask_deep_sea(3)
# # print(grid.landlake_arr)
#
# grid.build_cost_matrix()
# out = grid.neigbor_grid_output(flat_inner_matrix=True)
# # print(out)
# out.to_file('sample_out.nc')
#
# costs = out.uniform_weights(grid.landlake_arr)
# print(costs)
