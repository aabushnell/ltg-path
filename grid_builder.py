"""
5 September 2023

Class builder for grid object

Dijkstra's Implementation
"""

# import block
from __future__ import annotations
from itertools import product

import numpy as np
import xarray as xr
import h5netcdf
from haversine import haversine
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra

import cost_functions as cf
import grid_helpers as gh


class Grid:
    """
    Simple object for storing grid data in the form of 2D
    numpy arrays. Allows for accessing core array and
    subsetting grid (from upper left and lower right
    corners).
    """

    def __init__(self, cell_size: float, lat_start: float,
                 lon_start: float, lat_count: int, lon_count: int):
        if lat_count < 0 or lon_count < 0:
            raise ValueError
        self.cell_size = cell_size
        self.lat_start = lat_start
        self.lon_start = lon_start
        self.lat_count = lat_count
        self.lon_count = lon_count

        self.cell_count = self.lat_count * self.lon_count
        self._node_ids = (np.arange(0, self.cell_count)
                          .reshape(self.lat_count, self.lon_count))

    def __str__(self):
        return self.as_matrix().__str__()

    def __contains__(self, other: Grid) -> bool:
        return (self._node_ids[0, 0] <= other._node_ids[0, 0]
                and self._node_ids[-1, -1] >= other._node_ids[-1, -1])

    def node_id(self, index_y: int, index_x: int) -> int:
        return self._node_ids[index_y, index_x]

    def reset_ids(self) -> None:
        self._node_ids = (np.arange(0, self.cell_count)
                          .reshape(self.lat_count, self.lon_count))

    def as_matrix(self) -> np.ndarray[np.ndarray[int]]:
        return self._node_ids

    def subgrid(self, y1: int, x1: int, y2: int, x2: int) -> Grid:
        """
        Returns a subset of a grid based on two corners
        :param y1: y-coord of upper left corner
        :param x1: x-coord of upper left corner
        :param y2: y-coord of lower right corner
        :param x2: x-coord of upper left corner
        :return: new Grid object
        """
        lat, lon = gh.index_to_coord(y1, x1, self.cell_size,
                                     self.lat_start, self.lon_start)
        lat_count, lon_count = (y2 - y1 + 1, x2 - x1 + 1)
        new_grid = Grid(self.cell_size, lat, lon, lat_count, lon_count)
        new_grid._node_ids = self._node_ids[y1:y2 + 1, x1:x2 + 1]
        return new_grid


class NeighborCostMatrix:

    def __init__(self, grid_id: int,
                 cost_array: np.ndarray[np.ndarray[np.ndarray[float]]]):
        self.id = grid_id
        self.costs = cost_array
        self.dim = cost_array.shape[2]

    def __str__(self):
        return self.costs.__str__()

    @classmethod
    def from_file(cls, filename: str) -> NeighborCostMatrix:
        data_array_xarray = xr.open_dataarray(filename,
                                              engine='h5netcdf')
        grid_id = data_array_xarray.attrs['id']
        data_array_netcdf4 = data_array_xarray.to_numpy()
        data_array_xarray.close()
        return NeighborCostMatrix(grid_id, data_array_netcdf4)

    def to_file(self, filename: str) -> None:
        array_32 = np.float32(self.costs)
        (xr.DataArray(array_32, attrs={'id': self.id})
         .to_netcdf(filename, engine='h5netcdf'))

    def apply_weights(self,
                      orig_weights: np.ndarray[float],
                      dest_weights: np.ndarray[np.ndarray[float]]
                      ) -> np.ndarray[float]:
        if (orig_weights.shape[0] != self.dim
                or dest_weights.shape[1] != self.dim
                or dest_weights.shape[0] != 8):
            print(f"""Origin and destination weight arrays 
                      must be of shape ({self.dim}) and 
                      (8, {self.dim}) respectively.
                   """)
            raise ValueError

        orig = np.full((8, 1, self.dim), orig_weights)
        dest = dest_weights.reshape((8, self.dim, 1))
        return np.matmul(np.matmul(orig, self.costs), dest).flatten()

    def uniform_weights(self, elev_array: np.ndarray[np.ndarray[int]]
                        ) -> np.ndarray[float]:
        orig_values = gh.get_offset_subarray(elev_array, 0, 0).flatten()
        orig_weights = np.array([1 / (orig_values >= 0).sum()
                                 if n >= 0 else 0 for n in orig_values])

        dest_list = []
        for subarray in gh.all_outside_subarrays(elev_array):
            values = subarray.flatten()
            weights = np.array([1 / (values >= 0).sum()
                                if n >= 0 else 0 for n in values])
            dest_list.append(weights)
        dest_weights = np.array(dest_list)

        return self.apply_weights(orig_weights, dest_weights)


class DjikstraGrid:
    """
    Base object for all grids where pathfinding
    calculations are performed through Dijkstra's
    method. Contains a Grid object that stores the node
    id's of all grid cells. Also contains a cost matrix
    for movement between grid cells used in calculations.
    """

    def __init__(self, cell_size: float, lat_start: float,
                 lon_start: float, lat_count: int, lon_count: int,
                 cost_mat: dok_matrix | None = None):

        self.grid = Grid(cell_size, lat_start, lon_start, lat_count, lon_count)

        self.cost_mat = cost_mat

    @classmethod
    def from_grid(cls, grid: Grid,
                  cost_mat: dok_matrix | None = None) -> DjikstraGrid:
        return DjikstraGrid(grid.cell_size, grid.lat_start,
                            grid.lon_start, grid.lat_count,
                            grid.lon_count, cost_mat)

    def assign_cost_mat(self, cost_mat: dok_matrix) -> None:
        self.cost_mat = cost_mat

    # noinspection PyTupleAssignmentBalance,PyTypeChecker
    def eval_dijkstra(self, orig_ids: np.ndarray | None = None,
                      dest_ids: list[int] = None) -> np.ndarray:
        """
        Evaluates the cost of traversing from every node in
        orig_ids to every node in dest_ids. Object must have
        a defined cost matrix to evaluate this function.
        :param orig_ids: array of valid node ids for pathing
        to start from
        :param dest_ids: list of valid node ids for pathing
        to end in
        :return: matrix of costs between orig_id and
        dest_id pairs
        """
        if self.cost_mat is None:
            print('Cost Matrix for this grid not defined')
            raise ValueError

        if orig_ids is not None:
            dist_mat = dijkstra(self.cost_mat, indices=orig_ids)
        else:
            dist_mat = dijkstra(self.cost_mat)
        if dest_ids:
            return dist_mat[:, dest_ids]
        else:
            return dist_mat


class ElevationGrid(DjikstraGrid):
    DEEP_SEA_VALUE = -9999

    # ------------
    # constructors
    # ------------

    def __init__(self, grid_id: int,
                 elevation_array: np.ndarray[np.ndarray[int]],
                 cell_size: float, lat_start: float, lon_start: float,
                 landlake_array: np.ndarray[np.ndarray[int]],
                 deep_sea_depth: int | None = None,
                 river_array: np.ndarray[np.ndarray[int]] | None = None):

        # basic object values
        self.id = grid_id
        self.elev_arr = elevation_array
        self.landlake_arr = landlake_array

        self.array_count_y = self.elev_arr.shape[0]
        self.array_count_x = self.elev_arr.shape[1]

        # initialize DjikstraGrid
        super().__init__(cell_size, lat_start, lon_start,
                         self.array_count_y, self.array_count_x)

        # river handling
        if river_array is None:
            # initialize empty river array
            self.river_arr = np.zeros((self.grid.lat_count,
                                       self.grid.lon_count), dtype=np.int8)
        else:
            # check properly defined input array
            if (river_array.shape[0] != self.grid.lat_count
                    or river_array.shape[1] != self.grid.lon_count):
                print('River array must be same dimensions as base grid')
                raise ValueError
            self.river_arr = river_array

        # define deep sea points in landlake_arr
        if deep_sea_depth is not None:
            self.mask_deep_sea(depth=deep_sea_depth)

        self.land_nodes = self.grid.as_matrix()[self.elev_arr > 0]
        self.land_center_nodes = None
        self.grid_center = None
        self.elev_center = None

    # noinspection PyMethodOverriding
    # violates LSP but I don't care
    @classmethod
    def from_grid(cls, grid_id: int, grid: Grid,
                  elevation_array: np.ndarray[np.ndarray[int]],
                  landlake_array: np.ndarray[np.ndarray[int]],
                  deep_sea_depth: int | None = None,
                  river_array: np.ndarray[np.ndarray[int]] | None = None
                  ) -> ElevationGrid:
        new_grid = ElevationGrid(grid_id, elevation_array, grid.cell_size,
                                 grid.lat_start, grid.lon_start,
                                 landlake_array, deep_sea_depth, river_array)
        if (elevation_array.shape[0] != grid.lat_count
                or elevation_array.shape[1] != grid.lon_count):
            print('Elevation array dimensions do not match provided grid')
            raise ValueError
        new_grid.grid = grid
        return new_grid

    # ------------------
    # returning subgrids
    # ------------------

    def subgrid_from_mask(self, grid_id: int, mask: Grid,
                          reset_ids: bool = True,
                          deep_sea_depth: int | None = None
                          ) -> ElevationGrid:
        if (mask not in self.grid) or (self.grid.cell_size != mask.cell_size):
            print('Invalid masking grid, not proper subset of grid')
            raise ValueError

        elev_array = gh.mask_array(self.grid.as_matrix(), mask.as_matrix(),
                                   self.elev_arr)
        landlake_array = gh.mask_array(self.grid.as_matrix(), mask.as_matrix(),
                                       self.landlake_arr)
        river_array = gh.mask_array(self.grid.as_matrix(), mask.as_matrix(),
                                    self.river_arr)
        new_grid = ElevationGrid.from_grid(grid_id, mask, elev_array,
                                           landlake_array, deep_sea_depth,
                                           river_array)

        if reset_ids:
            new_grid.grid.reset_ids()
            return new_grid
        else:
            return new_grid

    def subgrid(self, grid_id: int, y1: int, x1: int, y2: int, x2: int,
                reset_ids: bool = True, deep_sea_depth: int | None = None
                ) -> ElevationGrid:
        mask = self.grid.subgrid(y1, x1, y2, x2)
        return self.subgrid_from_mask(grid_id, mask, reset_ids, deep_sea_depth)

    # ------------------
    # define cost matrix
    # ------------------

    def travel_cost(self, start_y: int, start_x: int,
                    end_y: int, end_x: int) -> float:
        start_elev = self.elev_arr[start_y, start_x]
        end_elev = self.elev_arr[end_y, end_x]

        start_terr = self.landlake_arr[start_y, start_x]
        end_terr = self.landlake_arr[end_y, end_x]

        if (self.river_arr[start_y, start_x]
                + self.river_arr[end_y, end_x] == 2):
            river_travel = True
        else:
            river_travel = False

        if start_elev != np.nan and end_elev != np.nan:
            start_coord = gh.index_to_coord(start_y, start_x,
                                            self.grid.cell_size,
                                            lat_start=self.grid.lat_start,
                                            lon_start=self.grid.lon_start,
                                            center=True)
            end_coord = gh.index_to_coord(end_y, end_x, self.grid.cell_size,
                                          lat_start=self.grid.lat_start,
                                          lon_start=self.grid.lon_start,
                                          center=True)
            distance = haversine(start_coord, end_coord)  # kms
            cost = cf.transport_method_cost(distance, start_elev,
                                            end_elev, start_terr,
                                            end_terr, river_travel,
                                            deep_sea_val=self.DEEP_SEA_VALUE)
            return cost
        else:
            raise ValueError

    def build_cost_matrix(self) -> dok_matrix:
        cost_mat = dok_matrix((self.grid.cell_count, self.grid.cell_count))

        for i, j in product(range(0, self.grid.lat_count),
                            range(0, self.grid.lon_count)):
            id_orig = self.grid.node_id(i, j)
            for i2, j2 in gh.get_valid_neighbors(i, j, 1, self.grid.lat_count,
                                                 self.grid.lon_count):
                cost_mat[id_orig, self.grid.node_id(i2, j2)] = (
                    self.travel_cost(i, j, i2, j2)
                )

        self.assign_cost_mat(cost_mat)
        return cost_mat

    # ------------------
    # modify grid values
    # ------------------

    def mask_deep_sea(self, depth: int,
                      deep_sea_val: int = DEEP_SEA_VALUE) -> None:
        for y, x in product(range(0, self.grid.lat_count),
                            range(0, self.grid.lon_count)):
            if self.landlake_arr[y, x] < 0:
                indices = gh.get_valid_neigbors_split(y, x, depth,
                                                      self.grid.lat_count,
                                                      self.grid.lon_count)
                points = self.landlake_arr[indices]
                if not (points > 0).any():
                    self.landlake_arr[y, x] = deep_sea_val

    def add_river_array(self,
                        river_array: np.ndarray[np.ndarray[int]]) -> None:
        if (self.grid.lat_count != river_array.shape[0]
                or self.grid.lon_count != river_array.shape[1]):
            print('River array must match dimensions of base grid')
            raise ValueError
        self.river_arr = river_array

    # -----------------------------------
    # travel cost for adjacent fine grids
    # -----------------------------------

    def distance_to_neighbors(self) -> np.ndarray:
        if self.land_center_nodes is None:
            print('Valid origin nodes not defined: calc_center must be called')
            raise ValueError
        center_ids = self.grid_center.flatten()

        return self.eval_dijkstra(orig_ids=center_ids)

    def calc_center(self) -> None:
        """
        Defines the center subgrid of the object's grid (assuming
        a 3x3 grid of subgrids). Requires the object to be defined
        on a square grid of values. Assigns the results as object
        variables.
        :return: None
        """

        if self.grid.lat_count != self.grid.lon_count:
            print('Elevation grids must be defined with symmetric dimensions')
            raise ValueError

        self.grid_center = gh.get_offset_subarray(self.grid.as_matrix(), 0, 0)
        self.elev_center = gh.get_offset_subarray(self.elev_arr, 0, 0)
        self.land_center_nodes = self.grid_center[self.elev_center > 0]

    def neigbor_grid_output(self, flat_inner_matrix: bool = False
                            ) -> NeighborCostMatrix:
        self.calc_center()
        full_grid = self.distance_to_neighbors()

        neighbors = [(i, j) for i in range(-1, 2) for j in range(-1, 2)
                     if i != 0 or j != 0]

        offsets = []
        for y_off, x_off in neighbors:
            origins = []
            for origin in full_grid:
                grid = origin.reshape(self.grid.lat_count, self.grid.lon_count)
                subgrid = gh.get_offset_subarray(grid, y_off, x_off)
                if flat_inner_matrix:
                    origins.append(subgrid.flatten())
                else:
                    origins.append(subgrid)
            offsets.append(origins)

        output = np.array(offsets)
        return NeighborCostMatrix(self.id, output)
