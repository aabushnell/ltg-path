"""
5 September 2023

Class builder for grid object

Dijkstra's Implementation
"""

# import block
import numpy as np

from haversine import haversine
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra

import cost_functions as cf
import grid_helpers as gh

class elevationGrid:

    def __init__(self, elevation_array, cell_size, starting_latitude, starting_longitude):
        self.elev_arr = elevation_array
        self.cell_size = cell_size
        self.lat_start = starting_latitude
        self.lon_start = starting_longitude

        self.lat_count = self.elev_arr.shape[0]
        self.lon_count = self.elev_arr.shape[1]
        self.cell_count = self.lat_count * self.lon_count

        self.node_ids = np.arange(0, self.cell_count).reshape(self.lat_count, self.lon_count)

    def get_travel_cost(self, start_y, start_x, end_y, end_x):
        """

        :param start_y:
        :param start_x:
        :param end_y:
        :param end_x:
        :return:
        """

        start_elev = self.elev_arr[start_y, start_x]
        end_elev = self.elev_arr[end_y, end_x]

        if start_elev != np.nan and end_elev != np.nan:

            start_coord = (self.lat_start - self.cell_size * (start_y + 1/2),
                           self.lon_start + self.cell_size * (start_x - 1/2))
            end_coord = (self.lat_start - self.cell_size * (end_y + 1/2),
                           self.lon_start + self.cell_size * (end_x - 1/2))

            distance = haversine(start_coord, end_coord)

            cost = cf.transport_method_cost(distance, start_elev, end_elev)
            return cost
        else: return 0

    def get_cost_matrix(self):
        """

        :return:
        """

        dist_mat = dok_matrix((self.cell_count, self.cell_count))

        for i in range(0, self.lon_count):
            for j in range(0, self.lat_count):
                id_orig = self.node_ids[i, j]
                for di, dj in gh.get_neighbors(i, j):
                    dist_mat[id_orig, self.node_ids[i + di, j + dj]] = self.get_travel_cost(i, j, i + di, j + dj)

        self.dist_mat = dist_mat
        return dist_mat





