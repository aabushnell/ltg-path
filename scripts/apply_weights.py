import numpy as np
import grid_builder as gb
import grid_helpers as gh
import file_reader as fr
import xarray as xr
import h5netcdf

from haversine import haversine

from itertools import product

CELL_SIZE = 1 / 12
COARSE_DIM = 24
X_MOD = 4320

path_weights = ('/Users/aaron/Documents/Work/Code/Pathfinding/data/'
                + 'weights.nc')

weights_full = gb.DataGrid(fr.read_nc(path_weights), CELL_SIZE, 90, 0)

path = ('/Users/aaron/Documents/Work/Code/GridViewer'
        + '/backend/app/data/RAW_COST_DATA_10/')

output = np.zeros((90, 180, 8))
outfile = '../weighted_costs_10.nc'

for y_coarse, x_coarse in product(range(9, 45), range(90)):
    if x_coarse < 13:
        x_coarse += 167
    else:
        x_coarse -= 13

    y_start = y_coarse * COARSE_DIM
    x_start = x_coarse * COARSE_DIM

    y_end = y_start + (COARSE_DIM - 1)
    x_end = x_start + (COARSE_DIM - 1)

    orig_weights = (weights_full
                    .subgrid(y_start, x_start, y_end, x_end)
                    .data
                    .flatten())

    dest_weights = np.zeros((8, 576))

    offsets = [(y, x) for (y, x)
               in product(range(-1, 2), range(-1, 2))
               if (y != 0 or x != 0)]

    for i, (y, x) in enumerate(offsets):
        y_offset_start = y_start + y * COARSE_DIM
        x_offset_start = (x_start + x * COARSE_DIM + X_MOD) % X_MOD

        y_offset_end = y_end + y * COARSE_DIM
        x_offset_end = (x_end + x * COARSE_DIM + X_MOD) % X_MOD

        weights = (weights_full
                   .subgrid(y_offset_start,
                            x_offset_start,
                            y_offset_end,
                            x_offset_end)
                   .data
                   .flatten())
        print(weights.shape)
        dest_weights[i, :] = weights

    cost_mat = (gb
                .BilateralCostMatrix
                .from_file(path + f'{y_coarse}/{x_coarse}.nc'))

    weighted_costs = cost_mat.apply_weights(orig_weights, dest_weights)

    print(y_coarse, x_coarse)
    print(weighted_costs)

    output[y_coarse, x_coarse, :] = weighted_costs

xr.DataArray(output).to_netcdf(outfile, engine='h5netcdf')
