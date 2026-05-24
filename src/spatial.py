from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import geopandas as gpd
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import transform

from .config import (
    ARTIFACT_DIR,
    CORE_TOWNS,
    EPSG_MODEL,
    EPSG_WEB,
    POI_LAYERS,
    ROAD_DISTANCE_FEATURES,
    SPATIAL_CACHE_PATH,
)

_TRANSFORMER_TO_MODEL = Transformer.from_crs(EPSG_WEB, EPSG_MODEL, always_xy=True)


def _require_layer(key: str):
    path, layer = POI_LAYERS[key]
    if not path.exists():
        raise FileNotFoundError(f"缺少必要 POI 圖資：{path.name}。請放入 POIs 資料夾後再啟動 app。")
    return path, layer


def _read_layer(key: str) -> gpd.GeoDataFrame:
    path, layer = _require_layer(key)
    gdf = gpd.read_file(path, layer=layer)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(EPSG_MODEL)
    if gdf.crs.to_epsg() != EPSG_MODEL:
        gdf = gdf.to_crs(EPSG_MODEL)
    gdf.geometry = gdf.geometry.map(lambda geom: transform(lambda x, y, z=None: (x, y), geom))
    return gdf


def _node_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), 3), round(float(y), 3))


def _build_road_graph() -> tuple[nx.Graph, np.ndarray]:
    roads = _read_layer("roads").explode(index_parts=False).copy()
    roads = roads[roads.geometry.geom_type == "LineString"].copy()
    graph = nx.Graph()
    node_ids: dict[tuple[float, float], int] = {}

    def get_node(coord) -> int:
        key = _node_key(coord[0], coord[1])
        if key not in node_ids:
            node_ids[key] = len(node_ids)
        return node_ids[key]

    for geom in roads.geometry:
        coords = list(geom.coords)
        for start, end in zip(coords[:-1], coords[1:]):
            u = get_node(start)
            v = get_node(end)
            if u == v:
                continue
            weight = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
            if weight > 0:
                graph.add_edge(u, v, weight=weight)

    coords_array = np.zeros((len(node_ids), 2), dtype=float)
    for coord, node_id in node_ids.items():
        coords_array[node_id] = coord
    return graph, coords_array


def _snap_points_to_nodes(gdf: gpd.GeoDataFrame, node_tree: cKDTree) -> list[int]:
    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    _, idx = node_tree.query(coords)
    return [int(i) for i in np.atleast_1d(idx)]


def _point_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()]).astype(float)


def build_and_save_spatial_cache(path=SPATIAL_CACHE_PATH) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    graph, node_coords = _build_road_graph()
    node_tree = cKDTree(node_coords)

    dist_maps: dict[str, dict[int, float]] = {}
    for key in ROAD_DISTANCE_FEATURES:
        poi_gdf = _read_layer(key)
        source_nodes = set(_snap_points_to_nodes(poi_gdf, node_tree))
        if not source_nodes:
            raise ValueError(f"{key} 圖層沒有可用點位，無法計算道路距離。")
        dist_maps[key] = nx.multi_source_dijkstra_path_length(graph, source_nodes, weight="weight")

    cache = {
        "graph": graph,
        "node_coords": node_coords,
        "dist_maps": dist_maps,
        "temple_coords": _point_coords(_read_layer("temple")),
        "stores_coords": _point_coords(_read_layer("stores")),
        "busstops_coords": _point_coords(_read_layer("busstops")),
        "medical_coords": _point_coords(_read_layer("medical")),
        "towns": _read_layer("towns"),
    }
    joblib.dump(cache, path)
    return _hydrate_cache(cache)


def _hydrate_cache(cache: dict[str, Any]) -> dict[str, Any]:
    cache = dict(cache)
    cache["node_tree"] = cKDTree(cache["node_coords"])
    cache["temple_tree"] = cKDTree(cache["temple_coords"])
    cache["stores_tree"] = cKDTree(cache["stores_coords"])
    cache["busstops_tree"] = cKDTree(cache["busstops_coords"])
    cache["medical_tree"] = cKDTree(cache["medical_coords"])
    return cache


@lru_cache(maxsize=1)
def load_or_build_spatial_cache() -> dict[str, Any]:
    if SPATIAL_CACHE_PATH.exists():
        return _hydrate_cache(joblib.load(SPATIAL_CACHE_PATH))
    return build_and_save_spatial_cache(SPATIAL_CACHE_PATH)


def _count_within(tree: cKDTree, x: float, y: float, radius: float = 500.0) -> int:
    return int(len(tree.query_ball_point([x, y], r=radius)))


def _core_zone(cache: dict[str, Any], point: Point) -> int:
    towns: gpd.GeoDataFrame = cache["towns"]
    matches = towns[towns.geometry.contains(point)]
    if matches.empty:
        return 0
    row = matches.iloc[0]
    if "core_zone" in row and pd.notna(row["core_zone"]):
        return int(row["core_zone"])
    town_name = str(row.get("TOWNNAME", "")).replace("臺中市", "")
    return int(town_name in CORE_TOWNS)


def compute_location_features(lon: float, lat: float, cache: dict[str, Any] | None = None) -> dict[str, Any]:
    cache = cache or load_or_build_spatial_cache()
    x, y = _TRANSFORMER_TO_MODEL.transform(float(lon), float(lat))
    point = Point(x, y)
    _, node_idx = cache["node_tree"].query([[x, y]])
    node = int(np.atleast_1d(node_idx)[0])

    features: dict[str, float] = {}
    details: dict[str, float | int] = {"x3826": float(x), "y3826": float(y)}
    for key, feature_name in ROAD_DISTANCE_FEATURES.items():
        dist = cache["dist_maps"][key].get(node)
        if dist is None:
            raise ValueError(f"目標點所在路網無法連通到 {key} 圖層，請改點鄰近道路位置。")
        dist = max(float(dist), 1.0)
        features[feature_name] = float(np.log(dist))
        details[f"dist_road_{key}_m"] = round(dist, 2)

    temple_dist = max(float(cache["temple_tree"].query([[x, y]])[0][0]), 1.0)
    stores_count = _count_within(cache["stores_tree"], x, y)
    bus_count = _count_within(cache["busstops_tree"], x, y)
    medical_count = _count_within(cache["medical_tree"], x, y)

    features["ln_dist_eucl_temple"] = float(np.log(temple_dist))
    features["ln_stores_500m"] = float(np.log1p(stores_count))
    features["ln_bus_stops_500m"] = float(np.log1p(bus_count))
    features["ln_medical_service_500m"] = float(np.log1p(medical_count))
    features["core_zone"] = float(_core_zone(cache, point))

    details.update(
        {
            "dist_eucl_temple_m": round(temple_dist, 2),
            "stores_500m": stores_count,
            "bus_stops_500m": bus_count,
            "medical_service_500m": medical_count,
            "core_zone": int(features["core_zone"]),
        }
    )
    return {"features": features, "details": details}
