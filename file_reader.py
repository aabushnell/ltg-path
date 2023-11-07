"""
TODO
"""
from typing import BinaryIO

import numpy as np


def read_int(f: BinaryIO) -> int:
    file_bytes = f.read(2)
    height = int.from_bytes(file_bytes, 'big', signed=True)
    return height


def read_elevation_full() -> np.ndarray[np.ndarray[int]]:
    lat = []
    path = '/Users/aaron/Documents/Work/Code/Pathfinding/data/ETOPO5.DAT'
    with open(path, 'rb') as f:
        for i in range(2160):
            lon = []
            for j in range(4320):
                height = read_int(f)
                lon.append(height)
            lat.append(lon)

    return np.array(lat)


def read_landlake_full() -> np.ndarray[np.ndarray[int]]:
    rows = []
    path = '/Users/aaron/Documents/Work/Code/Pathfinding/data/landlake.asc'
    with open(path, 'r') as f:
        for i in range(6):
            _ = f.readline()
        for i in range(2160):
            line_str = f.readline()
            line_arr = [int(n) for n in line_str.split(' ') if n != '\n']
            line_arr_fixed = line_arr[2160:4320] + line_arr[0:2160]
            rows.append(line_arr_fixed)

    return np.array(rows)
