"""
TODO
"""
from typing import BinaryIO

import numpy as np


def read_int(f: BinaryIO) -> int:
    file_bytes = f.read(2)
    height = int.from_bytes(file_bytes, 'big', signed=True)
    return height


def read_elevation_full() -> np.ndarray:
    lat = []
    with open('/Users/aaron/Desktop/ETOPO5.DAT', 'rb') as f:
        for i in range(2160):
            lon = []
            for j in range(4320):
                height = read_int(f)
                lon.append(height)
            lat.append(lon)

    return np.array(lat)
