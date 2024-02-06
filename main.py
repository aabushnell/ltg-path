import os
from itertools import product

import numpy as np

import grid_builder as gb
import grid_helpers as gh
import file_reader as fr
import rasterize_rivers as rr

CELL_SIZE = 1 / 12
COARSE_DIM = 24
# OUTPUT_FOLDER = '/Users/aaron/Documents/Work/Data/RAW_COST_DATA_SEA/'

ul = (52, 0)
ll = (30, 0)
ur = (52, 38)
lr = (30, 38)

coeffs = {
    'base': 1,  # cost per km (horizontal)
    'up': 1/120,  # cost per m (vertical)
    'down_mod': 1/360,  # cost per m (vertical)
    'down_steep_cutoff': -0.2125,  # angle
    'down_steep': -1/360,  # cost per m (vertical)
    'coastal_sea': 0.9,  # cost per km (horizontal) premium
    'deep_sea': 0.5,
    'loading': 3.6,  # == 1hr at 1m/s
    'river': 0.9,  # cost per km (horizontal) premium
}

uli = gh.coord_to_index(ul[0], ul[1], 1/12, 90, 0)
lri = gh.coord_to_index(lr[0], lr[1], 1/12, 90, 0)

elev_full = gb.DataGrid(fr.read_elevation_nc(), CELL_SIZE, 90, 0)
elev = elev_full.subgrid(uli[0], uli[1], lri[0], lri[1])
landlake_full = gb.DataGrid(fr.read_landlake_nc(), CELL_SIZE, 90, 0)
landlake = landlake_full.subgrid(uli[0], uli[1], lri[0], lri[1])
river_full = gb.DataGrid(fr.read_rivers_nc(), CELL_SIZE, 90, 0)
river = river_full.subgrid(uli[0], uli[1], lri[0], lri[1])

grid = gb.FineGrid(CELL_SIZE, 
                   ul[0], 
                   ul[1], 
                   coeffs, 
                   elev, 
                   landlake, 
                   river, 
                   3)

print(grid.grid.lat_count, grid.grid.lon_count)
grid.build_cost_matrix()
costs = grid.eval_dijkstra()

origin = (577, 149)
origin_coord = gh.index_to_coord(origin[0], origin[1], 1/12, 90, 0)
origin_ind = gh.coord_to_index(origin_coord[0], origin_coord[1], 1/12, ul[0], ul[1])
origin_id = grid.grid.node_id(origin_ind[0], origin_ind[1])
dest = (589, 171)
dest_coord = gh.index_to_coord(dest[0], dest[1], 1/12, 90, 0)
dest_ind = gh.coord_to_index(dest_coord[0], dest_coord[1], 1/12, ul[0], ul[1])
dest_id = grid.grid.node_id(dest_ind[0], dest_ind[1])

print(costs[origin_id, dest_id])

