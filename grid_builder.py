"""
5 September 2023

Class builder for grid object

Dijkstra's Implementation
"""

# import block
from itertools import product

import numpy as np
from haversine import haversine
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra

import cost_functions as cf
import grid_helpers as gh


class ElevationGrid:

    def __init__(self, elevation_array: np.ndarray[np.ndarray[int]],
                 cell_size: float, starting_latitude: float,
                 starting_longitude: float):
        self.elev_arr = elevation_array
        self.cell_size = cell_size
        self.lat_start = starting_latitude
        self.lon_start = starting_longitude

        self.lat_count = self.elev_arr.shape[0]
        self.lon_count = self.elev_arr.shape[1]
        self.cell_count = self.lat_count * self.lon_count

        self.node_ids = (np.arange(0, self.cell_count)
                         .reshape(self.lat_count, self.lon_count))
        self.cost_mat = None

    def travel_cost(self, start_y: int, start_x: int,
                    end_y: int, end_x: int) -> float:
        start_elev = self.elev_arr[start_y, start_x]
        end_elev = self.elev_arr[end_y, end_x]

        if start_elev != np.nan and end_elev != np.nan:
            start_coord = gh.index_to_coord(start_y, start_x, self.cell_size,
                                            lat_start=self.lat_start,
                                            lon_start=self.lon_start,
                                            center=True)
            end_coord = gh.index_to_coord(end_y, end_x, self.cell_size,
                                          lat_start=self.lat_start,
                                          lon_start=self.lon_start,
                                          center=True)
            distance = haversine(start_coord, end_coord)
            cost = cf.transport_method_cost(distance, start_elev, end_elev)
            return cost
        else:
            raise ValueError

    def cost_matrix(self) -> dok_matrix:
        cost_mat = dok_matrix((self.cell_count, self.cell_count))

        for i, j in product(range(0, self.lat_count),
                            range(0, self.lon_count)):
            id_orig = self.node_ids[i, j]
            for di, dj in gh.get_neighbors(i, j, self.lat_count,
                                           self.lon_count):
                cost_mat[id_orig, self.node_ids[i + di, j + dj]] = (
                    self.travel_cost(i, j, i + di, j + dj)
                )

        self.cost_mat = cost_mat
        return cost_mat

    # noinspection PyTupleAssignmentBalance
    def eval_dijkstra(self):
        trav_time, pred = dijkstra(self.cost_mat, directed=False,
                                   return_predecessors=True)
        return trav_time, pred
