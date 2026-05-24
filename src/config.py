from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = APP_ROOT / "Taichung_rental_houses_v4.gpkg"
POI_DIR = APP_ROOT / "POIs"
ARTIFACT_DIR = APP_ROOT / "artifacts"
MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "lightgbm_rent_model.joblib"
SPATIAL_CACHE_PATH = ARTIFACT_DIR / "spatial_cache.joblib"

EPSG_MODEL = 3826
EPSG_WEB = 4326
SEED = 42
ML_N = 30_000
KNN_K = 20

BASE_X = [
    "pet_friendly", "limited", "deposit_months", "mgmt_fee", "water_fee",
    "area_pings", "parking", "apartment", "elevator_building",
    "ln_dist_road_railsta", "ln_dist_road_mrt", "ln_dist_road_ubike",
    "ln_dist_road_highway", "ln_dist_road_park", "ln_dist_road_school",
    "ln_dist_eucl_temple", "ln_stores_500m", "ln_bus_stops_500m",
    "ln_medical_service_500m", "core_zone", "air_conditioner", "laundry",
    "sum_equip_idx",
]
INTER_VARS = ["inter_equip", "inter_apt", "inter_elev", "inter_core"]
SPATIAL_VARS = ["Wy", "W_pet_friendly"]
FEATURE_COLUMNS = BASE_X + INTER_VARS + SPATIAL_VARS

CONTINUOUS_VARS = [
    "area_pings", "ln_dist_road_railsta", "ln_dist_road_mrt", "ln_dist_road_ubike",
    "ln_dist_road_highway", "ln_dist_road_park", "ln_dist_road_school",
    "ln_dist_eucl_temple", "ln_stores_500m", "ln_bus_stops_500m",
    "ln_medical_service_500m", "sum_equip_idx",
]

USER_CONTINUOUS = ["area_pings", "deposit_months", "mgmt_fee", "water_fee", "sum_equip_idx"]
USER_BINARY = [
    "pet_friendly", "limited", "parking", "apartment", "elevator_building",
    "air_conditioner", "laundry",
]

POI_LAYERS = {
    "railsta": (POI_DIR / "Taichung_rail_stations.gpkg", "rail_station"),
    "mrt": (POI_DIR / "Taichung_MRT.gpkg", "Taichung_MRT"),
    "ubike": (POI_DIR / "Taichung_youbikes.gpkg", "youbike20"),
    "highway": (POI_DIR / "Taichung_highway_inters.gpkg", "taichung_highway_inters"),
    "park": (POI_DIR / "Taichung_parks.gpkg", "parks"),
    "school": (POI_DIR / "Taichung_schools.gpkg", "schools"),
    "temple": (POI_DIR / "Taichung_temples.gpkg", "temples"),
    "stores": (POI_DIR / "Taichung_stores.gpkg", "stores"),
    "busstops": (POI_DIR / "Taichung_busstops.gpkg", "busstops"),
    "medical": (POI_DIR / "Taichung_medical_service.gpkg", "hospital_done_0924"),
    "towns": (POI_DIR / "taichung_town_joined_2.gpkg", "taichung_town_joined_2"),
    "roads": (POI_DIR / "112Taichung_road_network.gpkg", "112Taichung_road_network"),
}

ROAD_DISTANCE_FEATURES = {
    "railsta": "ln_dist_road_railsta",
    "mrt": "ln_dist_road_mrt",
    "ubike": "ln_dist_road_ubike",
    "highway": "ln_dist_road_highway",
    "park": "ln_dist_road_park",
    "school": "ln_dist_road_school",
}

CORE_TOWNS = {"東區", "西區", "南區", "北區", "中區", "西屯區", "北屯區", "南屯區"}

FEATURE_DESCRIPTIONS = {
    "pet_friendly": "是否可養寵物，1=可，0=不可。",
    "limited": "租屋限制條件，1=有限制，0=未標示限制。",
    "deposit_months": "押金月數。",
    "mgmt_fee": "管理費。",
    "water_fee": "水費。",
    "area_pings": "出租坪數。",
    "parking": "是否提供停車位。",
    "apartment": "是否為公寓類型。",
    "elevator_building": "是否為電梯大樓。",
    "ln_dist_road_railsta": "至最近火車站的道路距離取自然對數。",
    "ln_dist_road_mrt": "至最近捷運站出口的道路距離取自然對數。",
    "ln_dist_road_ubike": "至最近 YouBike 站的道路距離取自然對數。",
    "ln_dist_road_highway": "至最近交流道的道路距離取自然對數。",
    "ln_dist_road_park": "至最近公園的道路距離取自然對數。",
    "ln_dist_road_school": "至最近學校的道路距離取自然對數。",
    "ln_dist_eucl_temple": "至最近寺廟的直線距離取自然對數。",
    "ln_stores_500m": "500 公尺環域內商店數量取 ln(count+1)。",
    "ln_bus_stops_500m": "500 公尺環域內公車站數量取 ln(count+1)。",
    "ln_medical_service_500m": "500 公尺環域內醫療診所數量取 ln(count+1)。",
    "core_zone": "是否位於核心商業區。",
    "air_conditioner": "是否提供冷氣。",
    "laundry": "是否提供洗衣設備。",
    "sum_equip_idx": "家具、其他家電與影音網路設備指標總和。",
    "inter_equip": "pet_friendly 與 sum_equip_idx 的交互項。",
    "inter_apt": "pet_friendly 與 apartment 的交互項。",
    "inter_elev": "pet_friendly 與 elevator_building 的交互項。",
    "inter_core": "pet_friendly 與 core_zone 的交互項。",
    "Wy": "目標點鄰近 20 筆租屋樣本 ln_rent 的空間滯後平均。",
    "W_pet_friendly": "目標點鄰近 20 筆租屋樣本 pet_friendly 的空間滯後平均。",
}
