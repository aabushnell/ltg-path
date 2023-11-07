import grid_helpers as gh

from itertools import product
import json

offset = 0.00027778
cellsize = 2

features = []
for y, x in product(range(90), range(180)):

    cell_id = y * 180 + x

    center_lat, center_lon = gh.index_to_coord(y, x, cellsize, center=True)

    lat, lon = gh.index_to_coord(y, x, cellsize)
    ul = [max(lon - offset, 0),
          min(lat + offset, 90)]
    ll = [max(lon - offset, 0),
          max(lat - offset - cellsize, -90)]
    lr = [min(lon + offset + cellsize, 360),
          max(lat - offset - cellsize, -90)]
    ur = [min(lon + offset + cellsize, 360),
          min(lat + offset, 90)]
    d = {
            "type": "Feature",
            "geometry":
                {
                    "type": "Polygon",
                    "coordinates":
                        [[
                            ll,
                            lr,
                            ur,
                            ul,
                            ll
                        ]]
                },
            "properties":
                {
                    "id": f"{y}.{x}",
                    "index_y": y,
                    "index_x": x,
                    "cellsize": cellsize,
                    "center_lat": center_lat,
                    "center_lon": center_lon
                },
            "id": cell_id
        }
    features.append(d)

gj = {"type": "FeatureCollection", "features": features}
print(gj)
with open("coarse_grid.json", "w") as outfile:
    json.dump(gj, outfile)