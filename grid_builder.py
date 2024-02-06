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

    # def join(self, other: Grid, side: str) -> Grid:
    #
    #     match side:
    #         case 'l':
    #
    #         case 'r':
    #         case 't':
    #         case 'b':
    #         case _:
    #             print('side value must be one of l, r, t, or b')
    #             raise ValueError


class DataGrid(Grid):

    def __init__(self, data: np.ndarray[np.ndarray[int]],
                 cell_size: float, lat_start: float, lon_start: float):

        self.data = data

        lat_count = data.shape[0]
        lon_count = data.shape[1]

        super().__init__(cell_size, lat_start, lon_start, lat_count, lon_count)

    def __str__(self):
        return self.data.__str__()

    def loc(self, y: int, x: int) -> int:
        return self.data[y, x]

    def from_indices(self, indices: tuple[list[int], list[int]]) -> np.ndarray:
        return self.data[indices]

    def assign(self, y: int, x: int, val: int) -> None:
        self.data[y, x] = val

    def as_matrix(self) -> np.ndarray[np.ndarray[int]]:
        return self.data

    def subgrid(self, y1: int, x1: int, y2: int, x2: int) -> DataGrid:
        data = self.data[y1:(y2 + 1), x1:(x2 + 1)]
        lat, lon = gh.index_to_coord(y1, x1, self.cell_size,
                                     self.lat_start, self.lon_start)

        return DataGrid(data, self.cell_size, lat, lon)

    def join(self, other: DataGrid, side: str) -> DataGrid:
        match side:
            case 'b':
                data_joined = np.concatenate((self.data, other.data), axis=0)
                lat_start = self.lat_start
                lon_start = self.lon_start
            case 'r':
                data_joined = np.concatenate((self.data, other.data), axis=1)
                lat_start = self.lat_start
                lon_start = self.lon_start
            case 't':
                data_joined = np.concatenate((other.data, self.data), axis=0)
                lat_start = other.lat_start
                lon_start = other.lon_start
            case 'l':
                data_joined = np.concatenate((other.data, self.data), axis=1)
                lat_start = other.lat_start
                lon_start = other.lon_start
            case _:
                print('side value must be one of l, r, t, or b')
                raise ValueError

        return DataGrid(data_joined, self.cell_size, lat_start, lon_start)


class BilateralCostMatrix:

    def __init__(self, grid_id: int,
                 cost_array: np.ndarray[np.ndarray[np.ndarray[float]]]):
        self.id = grid_id
        self.costs = cost_array
        self.dim = cost_array.shape[2]

    def __str__(self):
        return self.costs.__str__()

    @classmethod
    def from_file(cls, filename: str) -> BilateralCostMatrix:
        data_array_xarray = xr.open_dataarray(filename,
                                              engine='h5netcdf')
        grid_id = data_array_xarray.attrs['id']
        data_array_netcdf4 = data_array_xarray.to_numpy()
        data_array_xarray.close()
        return BilateralCostMatrix(grid_id, data_array_netcdf4)

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


class DjikstraGrid:
    """
    Base object for all grids where pathfinding
    calculations are performed through Dijkstra's
    method. Contains a Grid object that stores the node
    id's of all grid cells. Also contains a cost matrix
    for movement between grid cells used in calculations.
    """

    def __init__(self,
                 cell_size: float,
                 lat_start: float,
                 lon_start: float,
                 lat_count: int,
                 lon_count: int,
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
    def eval_dijkstra(self,
                      orig_ids: np.ndarray | None = None,
                      dest_ids: list[int] | None = None) -> np.ndarray:
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

    # noinspection PyTupleAssignmentBalance,PyTypeChecker
    def eval_dijkstra_pred(self,
                           orig_ids: list[int] | None = None,
                           dest_ids: list[int] | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
        if self.cost_mat is None:
            print('Cost Matrix for this grid not defined')
            raise ValueError

        if orig_ids is not None:
            dist_mat, pred_mat = dijkstra(self.cost_mat,
                                          indices=orig_ids,
                                          return_predecessors=True)
        else:
            dist_mat, pred_mat = dijkstra(self.cost_mat,
                                          return_predecessors=True)
        if dest_ids:
            return dist_mat[:, dest_ids], pred_mat
        else:
            return dist_mat, pred_mat


class BilateralCostGrid(DjikstraGrid):
    DEEP_SEA_VALUE = -9999

    # ------------
    # constructors
    # ------------

    def __init__(self, grid_id: int,
                 cell_size: float, lat_start: float, lon_start: float,
                 elevation_array: DataGrid,
                 landlake_array: DataGrid,
                 river_array: DataGrid | None = None,
                 deep_sea_depth: int | None = None):

        self.id = grid_id
        self.elev_arr = elevation_array
        self.landlake_arr = landlake_array

        self.array_count_y = self.elev_arr.lat_count
        self.array_count_x = self.elev_arr.lon_count

        super().__init__(cell_size, lat_start, lon_start,
                         self.array_count_y, self.array_count_x)

        if river_array is None:
            # initialize empty river array
            zero_arr = np.zeros((self.grid.lat_count,
                                 self.grid.lon_count),
                                dtype=np.int8)
            self.river_arr = DataGrid(zero_arr,
                                      self.grid.cell_size,
                                      self.grid.lat_start,
                                      self.grid.lon_start)
        else:
            # check properly defined input array
            if (river_array.lat_count != self.grid.lat_count
                    or river_array.lon_count != self.grid.lon_count):
                print('River array must be same dimensions as base grid')
                raise ValueError
            self.river_arr = river_array

        # define deep sea points in landlake_arr
        if deep_sea_depth is not None:
            self.mask_deep_sea(depth=deep_sea_depth)
        self.deep_sea_depth = deep_sea_depth

        self.grid_center = gh.get_offset_subarray(self.grid.as_matrix(), 0, 0)

    # noinspection PyMethodOverriding
    @classmethod
    def from_grid(cls, grid_id: int, grid: Grid,
                  elevation_array: DataGrid,
                  landlake_array: DataGrid,
                  deep_sea_depth: int | None = None,
                  river_array: DataGrid | None = None
                  ) -> BilateralCostGrid:

        new_grid = BilateralCostGrid(grid_id,
                                     grid.cell_size,
                                     grid.lat_start,
                                     grid.lon_start,
                                     elevation_array,
                                     landlake_array,
                                     river_array,
                                     deep_sea_depth)

        if (elevation_array.lat_count != grid.lat_count
                or elevation_array.lon_count != grid.lon_count):
            print('Elevation array dimensions do not match provided grid')
            raise ValueError
        new_grid.grid = grid
        return new_grid

    def subgrid(self, grid_id: int, y1: int, x1: int, y2: int, x2: int,
                reset_ids: bool = True
                ) -> BilateralCostGrid:

        elev_arr = self.elev_arr.subgrid(y1, x1, y2, x2)
        landlake_arr = self.landlake_arr.subgrid(y1, x1, y2, x2)
        river_arr = self.river_arr.subgrid(y1, x1, y2, x2)

        subgrid = BilateralCostGrid(grid_id,
                                    self.grid.cell_size,
                                    self.grid.lat_start,
                                    self.grid.lon_start,
                                    elev_arr,
                                    landlake_arr,
                                    river_arr,
                                    None)

        if reset_ids:
            subgrid.grid.reset_ids()
        return subgrid

    def is_center_land(self) -> bool:
        landlake_matrix = self.landlake_arr.as_matrix()
        landlake_center = gh.get_offset_subarray(landlake_matrix, 0, 0)
        return (landlake_center >= 0).any()

    # ------------------
    # modify grid values
    # ------------------

    def mask_deep_sea(self, depth: int,
                      deep_sea_val: int = DEEP_SEA_VALUE) -> None:
        for y, x in product(range(0, self.grid.lat_count),
                            range(0, self.grid.lon_count)):
            if self.landlake_arr.loc(y, x) < 1:
                indices = gh.get_valid_neigbors_split(y, x, depth,
                                                      self.grid.lat_count,
                                                      self.grid.lon_count)
                points = self.landlake_arr.from_indices(indices)
                if not (points > 0).any():
                    self.landlake_arr.assign(y, x, deep_sea_val)

    def add_river_array(self, river_array: DataGrid) -> None:
        if (self.grid.lat_count != river_array.lat_count
                or self.grid.lon_count != river_array.lon_count):
            print('River array must match dimensions of base grid')
            raise ValueError
        self.river_arr = river_array

    # ------------------
    # define cost matrix
    # ------------------

    def travel_distance(self, start_y: int, start_x: int,
                        end_y: int, end_x: int):
        start_coord = gh.index_to_coord(start_y, start_x,
                                        self.grid.cell_size,
                                        lat_start=self.grid.lat_start,
                                        lon_start=self.grid.lon_start,
                                        center=True, normalize_lon=True)
        end_coord = gh.index_to_coord(end_y, end_x, self.grid.cell_size,
                                      lat_start=self.grid.lat_start,
                                      lon_start=self.grid.lon_start,
                                      center=True, normalize_lon=True)
        return haversine(start_coord, end_coord)  # kms

    def travel_cost(self, start_y: int, start_x: int,
                    end_y: int, end_x: int) -> float:
        start_elev = self.elev_arr.loc(start_y, start_x)
        end_elev = self.elev_arr.loc(end_y, end_x)

        start_terr = self.landlake_arr.loc(start_y, start_x)
        end_terr = self.landlake_arr.loc(end_y, end_x)

        if (self.river_arr.loc(start_y, start_x)
                + self.river_arr.loc(end_y, end_x) == 2):
            river_travel = True
        else:
            river_travel = False

        if start_elev != np.nan and end_elev != np.nan:
            distance = self.travel_distance(start_y, start_x, end_y, end_x)
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
            for i2, j2 in gh.get_valid_neighbors(i, j, 1,
                                                 self.grid.lat_count,
                                                 self.grid.lon_count):
                cost_mat[id_orig, self.grid.node_id(i2, j2)] = (
                    self.travel_cost(i, j, i2, j2)
                )

        self.assign_cost_mat(cost_mat)
        return cost_mat

    # -----------------------------------
    # travel cost for adjacent fine grids
    # -----------------------------------

    def bilateral_matrix_output(self,
                                flat_inner_matrix: bool = False
                                ) -> BilateralCostMatrix:

        center_ids = self.grid_center.flatten()
        full_grid = self.eval_dijkstra(orig_ids=center_ids)

        neighbors = [(i, j) for i in range(-1, 2)
                     for j in range(-1, 2)
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
        return BilateralCostMatrix(self.id, output)


class BilateralCostGridSea(BilateralCostGrid):

    def __init__(self, grid_id: int,
                 cell_size: float, lat_start: float, lon_start: float,
                 elevation_array: DataGrid,
                 landlake_array: DataGrid):

        super().__init__(grid_id, cell_size, lat_start, lon_start,
                         elevation_array, landlake_array)

    # def mask_land(self):
    #     for y, x in product(range(0, self.grid.lat_count),
    #                         range(0, self.grid.lon_count)):
    #         if self.landlake_arr.loc(y, x) > 0:
    #             indices = gh.get_valid_neigbors_split(y, x, depth,
    #                                                   self.grid.lat_count,
    #                                                   self.grid.lon_count)
    #             points = self.landlake_arr.from_indices(indices)
    #             if not (points > 0).any():
    #                 self.landlake_arr.assign(y, x, deep_sea_val)

    def travel_cost(self, start_y: int, start_x: int,
                    end_y: int, end_x: int) -> float:
        start_terr = self.landlake_arr.loc(start_y, start_x)
        end_terr = self.landlake_arr.loc(end_y, end_x)

        if start_terr != np.nan and end_terr != np.nan:
            distance = self.travel_distance(start_y, start_x, end_y, end_x)
            cost = cf.high_seas_cost(distance, start_terr, end_terr)
            return cost
        else:
            raise ValueError


class FineGrid(DjikstraGrid):
    DEEP_SEA_VALUE = -2

    def __init__(self, 
                 cell_size: float,
                 lat_start: float,
                 lon_start: float,
                 cost_coeffs: dict[str, float],
                 elevation_array: DataGrid,
                 landlake_array: DataGrid,
                 river_array: DataGrid | None = None,
                 deep_sea_depth: int | None = None):

        self.elev_arr = elevation_array
        self.landlake_arr = landlake_array

        self.cost_coeffs = cost_coeffs

        self.array_count_y = self.elev_arr.lat_count
        self.array_count_x = self.elev_arr.lon_count

        super().__init__(cell_size,
                         lat_start,
                         lon_start,
                         self.array_count_y, 
                         self.array_count_x)

        if river_array is None:
            # initialize empty river array
            zero_arr = np.zeros((self.grid.lat_count,
                                 self.grid.lon_count),
                                dtype=np.int8)
            self.river_arr = DataGrid(zero_arr,
                                      self.grid.cell_size,
                                      self.grid.lat_start,
                                      self.grid.lon_start)
        else:
            # check properly defined input array
            if (river_array.lat_count != self.grid.lat_count
                    or river_array.lon_count != self.grid.lon_count):
                print('River array must be same dimensions as base grid')
                raise ValueError
            self.river_arr = river_array

        # define deep sea points in landlake_arr
        if deep_sea_depth is not None:
            self.mask_deep_sea(depth=deep_sea_depth)
        self.deep_sea_depth = deep_sea_depth

    def mask_deep_sea(self,
                      depth: int,
                      deep_sea_val: int = DEEP_SEA_VALUE) -> None:

        for y, x in product(range(0, self.grid.lat_count),
                            range(0, self.grid.lon_count)):
            if self.landlake_arr.loc(y, x) < 1: # 1 => Land
                indices = gh.get_valid_neigbors_split(y, x, depth,
                                                      self.grid.lat_count,
                                                      self.grid.lon_count)
                nbr_points = self.landlake_arr.from_indices(indices)
                if not (nbr_points > 0).any(): # 0 => Lake, -1 => Sea 
                    self.landlake_arr.assign(y, x, deep_sea_val)

    def travel_distance(self,
                        start_y: int, 
                        start_x: int,
                        end_y: int, 
                        end_x: int):

        start_coord = gh.index_to_coord(start_y, start_x,
                                        self.grid.cell_size,
                                        lat_start=self.grid.lat_start,
                                        lon_start=self.grid.lon_start,
                                        center=True, normalize_lon=True)
        end_coord = gh.index_to_coord(end_y, end_x, self.grid.cell_size,
                                      lat_start=self.grid.lat_start,
                                      lon_start=self.grid.lon_start,
                                      center=True, normalize_lon=True)
        return haversine(start_coord, end_coord)  # kms

    def travel_cost(self, 
                    start_y: int, 
                    start_x: int,
                    end_y: int, 
                    end_x: int) -> float:

        start_elev = self.elev_arr.loc(start_y, start_x)
        end_elev = self.elev_arr.loc(end_y, end_x)

        start_terr = self.landlake_arr.loc(start_y, start_x)
        end_terr = self.landlake_arr.loc(end_y, end_x)

        if (self.river_arr.loc(start_y, start_x)
                + self.river_arr.loc(end_y, end_x) == 2):
            river_travel = True
        else:
            river_travel = False

        if start_elev != np.nan and end_elev != np.nan:
            distance = self.travel_distance(start_y, start_x, end_y, end_x)
            cost = cf.transport_method_cost(distance, 
                                            start_elev,
                                            end_elev, 
                                            start_terr,
                                            end_terr, 
                                            river_travel,
                                            deep_sea_val=self.DEEP_SEA_VALUE,
                                            coeffs = self.cost_coeffs,
                                            allow_deep = True,
                                            allow_lake = False)
            return cost
        else:
            raise ValueError

    def build_cost_matrix(self) -> dok_matrix:
        cost_mat = dok_matrix((self.grid.cell_count, self.grid.cell_count))

        for i, j in product(range(0, self.grid.lat_count),
                            range(0, self.grid.lon_count)):
            id_orig = self.grid.node_id(i, j)
            for i2, j2 in gh.get_valid_neighbors(i, j, 1,
                                                 self.grid.lat_count,
                                                 self.grid.lon_count,
                                                 overflow=True):
                cost_mat[id_orig, self.grid.node_id(i2, j2)] = (
                    self.travel_cost(i, j, i2, j2)
                )

        self.assign_cost_mat(cost_mat)
        return cost_mat


class CoarseGrid(DjikstraGrid):
    EMBARK_COST = 50

    def __init__(self,
                 cell_size: int,
                 lat_start: float,
                 lon_start: float,
                 lat_count: int,
                 lon_count: int,
                 bilateral_costs_land: np.ndarray,
                 bilateral_costs_sea: np.ndarray,
                 land_cells: list[int],
                 coast_cells: list[int],
                 sea_cells: list[int],
                 impassable_cells: list[int]):

        super().__init__(cell_size,
                         lat_start,
                         lon_start,
                         lat_count,
                         lon_count)

        self.land_cells = land_cells
        self.coast_cells = coast_cells
        self.sea_cells = sea_cells
        self.impass_cells = impassable_cells

        self.cost_mat = self.build_cost_mat(bilateral_costs_land,
                                            bilateral_costs_sea)

    def build_cost_mat(self, bilateral_costs_land: np.ndarray,
                       bilateral_costs_sea: np.ndarray) -> dok_matrix:
        cost_mat = dok_matrix((self.grid.cell_count * 2,
                               self.grid.cell_count * 2))

        for y, x in product(range(0, self.grid.lat_count),
                            range(0, self.grid.lon_count)):

            grid_id = (360 // self.grid.cell_size) * y + x
            if grid_id in self.land_cells or grid_id in self.coast_cells:
                adj_costs = bilateral_costs_land[y, x]
                id_offset = 0
                if (adj_costs != 0).any():
                    for i, (y2, x2) in enumerate(
                            gh.get_valid_neighbors(y, x, 1,
                                                   self.grid.lat_count,
                                                   self.grid.lon_count,
                                                   overflow=True)
                    ):
                        origin_id = self.grid.node_id(y, x) + id_offset
                        dest_id = self.grid.node_id(y2, x2) + id_offset
                        if (origin_id not in self.impass_cells
                                and dest_id not in self.impass_cells):
                            cost_mat[origin_id, dest_id] = adj_costs[i]
            if grid_id in self.sea_cells or grid_id in self.coast_cells:
                adj_costs = bilateral_costs_sea[y, x]
                id_offset = 16200
                if (adj_costs != 0).any():
                    for i, (y2, x2) in enumerate(
                            gh.get_valid_neighbors(y, x, 1,
                                                   self.grid.lat_count,
                                                   self.grid.lon_count,
                                                   overflow=True)
                    ):
                        origin_id = self.grid.node_id(y, x) + id_offset
                        dest_id = self.grid.node_id(y2, x2) + id_offset
                        if (origin_id not in self.impass_cells
                                and dest_id not in self.impass_cells):
                            # if y == 27 and x == 7:
                            #     print('found')
                            #     print(origin_id, dest_id, adj_costs[i])
                            cost_mat[origin_id, dest_id] = adj_costs[i]

        for node in self.coast_cells:
            if node not in self.impass_cells:
                cost_mat[node, node + 16200] = self.EMBARK_COST
                cost_mat[node + 16200, node] = self.EMBARK_COST

        return cost_mat

    def get_path(self, origin_id, dest_id):
        dist, pred = self.eval_dijkstra_pred([origin_id], [dest_id])
        path = []
        next_id = dest_id
        while pred[0, next_id] != origin_id:
            path.append(next_id)
            next_id = pred[0, next_id]
        path.append(next_id)
        path.append(origin_id)
        return path
