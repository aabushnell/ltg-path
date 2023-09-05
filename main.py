import numpy as np
from timeit import timeit

import grid_builder as gb

def read_int(f):
    bytes = f.read(2)
    height = int.from_bytes(bytes, 'big', signed=True)
    return height

lat = []
with open('/Users/aaron/Desktop/ETOPO5.DAT', 'rb') as f:
    for i in range(2160):
        lon = []
        for j in range(4320):
            height = read_int(f)
            lon.append(height)
        lat.append(lon)

arr_full = np.array(lat)

# arr = arr_full[490:514, 20:44]
arr = arr_full[490:495, 20:25]

# costs:
cost_array = [1, 1, 1]

grid = gb.elevationGrid(arr, cell_size=(1/12), starting_latitude=90, starting_longitude=0)

print(arr)
print(grid.node_ids)
print(grid.get_cost_matrix())
