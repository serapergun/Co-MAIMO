"""Assign passenger-accessible SUMO edges to administrative districts.

Requires geopandas + sumolib. Maximum line-length intersection is preferred
over a midpoint-only assignment for boundary-crossing edges.
"""
from __future__ import annotations
import geopandas as gpd
from shapely.geometry import LineString
import sumolib

def map_edges_to_districts(net_file, district_file, district_name_col):
    net = sumolib.net.readNet(net_file)
    districts = gpd.read_file(district_file)
    if district_name_col not in districts.columns:
        raise KeyError(f"Missing district field: {district_name_col}")
    if districts.crs is None:
        raise ValueError("District boundary CRS is missing.")

    rows = []
    for edge in net.getEdges():
        if "passenger" not in edge.getAllowed():
            continue
        shape = edge.getShape()
        if len(shape) < 2:
            continue
        lonlat = [net.convertXY2LonLat(x, y) for x, y in shape]
        geom = gpd.GeoSeries([LineString(lonlat)], crs="EPSG:4326").to_crs(districts.crs).iloc[0]
        candidates = districts[districts.intersects(geom)].copy()
        if candidates.empty:
            continue
        candidates["_intersection_length"] = candidates.geometry.intersection(geom).length
        hit = candidates.loc[candidates["_intersection_length"].idxmax()]
        rows.append({
            "edge_id": edge.getID(),
            "district": hit[district_name_col],
            "edge_length_m": edge.getLength(),
            "intersection_length": float(hit["_intersection_length"]),
        })
    return gpd.pd.DataFrame(rows)
