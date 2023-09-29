import numpy as np
import geopandas as gpd
import rasterio.transform
from rasterio import features
from scipy.sparse import dok_matrix


def read_rivers_raster(cellsize):
    rivers = gpd.read_file(
        '/Users/aaron/Desktop/ne_10m_rivers_lake_centerlines.json')

    # print(rivers.geometry)

    shape = [(r, 1) for r in rivers.geometry]

    transform = rasterio.transform.from_origin(0, 90, cellsize, cellsize)

    mat = features.rasterize(shapes=shape,
                             out_shape=(2160, 4320),
                             fill=0,
                             transform=transform,
                             all_touched=True)

    return mat
