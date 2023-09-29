import numpy as np

import grid_builder as gb
import grid_helpers as gh
import file_reader as fr
import rasterize_rivers as rr

CELL_SIZE = 1/12

elev_full = fr.read_elevation_full()
river_full = rr.read_rivers_raster(cellsize=CELL_SIZE)
full_elev_grid = gb.ElevationGrid(elev_full, cell_size=CELL_SIZE,
                                  lat_start=0, lon_start=0,
                                  river_array=river_full)

# print(full_elev_grid.grid._node_ids.shape)

# full_elev_grid.mask_deep_sea(3)

COARSE_DIM = 3
# Y_START = 626
# X_START = 1466

Y_START = 490
X_START = 20

y_end = Y_START + COARSE_DIM*3 - 1
x_end = X_START + COARSE_DIM*3 - 1

grid = full_elev_grid.subgrid(Y_START, X_START, y_end, x_end)

print(grid.grid)
print(grid.elev_arr)
print(grid.river_arr)


# grid.build_cost_matrix()
# print(grid.neigbor_grid_output(flat_inner_matrix=True))
