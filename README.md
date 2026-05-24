---
title: Taichung Rental LightGBM
emoji: 🏠
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
---

# 臺中市租屋每坪租金預測 WebApp

本專案是一個以 Python、Solara、leafmap 與 LightGBM 建立的臺中市租屋每坪租金預測 WebApp，可部署於 HuggingFace Spaces。模型使用 `Taichung_rental_houses_v4.gpkg` 的租屋點位與屬性資料，依據課堂 notebook 的 F3_SPATIAL 特徵集訓練 LightGBM 模型，預測目標為 `ln_rent`，並在介面中還原為「元 / 坪」。

## 主要功能

- 首頁儀表板：呈現模型表現指標、特徵變數說明、LightGBM 特徵重要性與 SHAP 平均絕對貢獻度。
- 地圖預測頁：以 OpenStreetMap 與租屋點位圖層顯示臺中市租屋樣本，使用紅色漸層呈現 `ln_rent` 大小。
- 互動式座標選取：使用者可在地圖上點選目標房屋位置，或拖曳目標標記調整座標。
- 空間區位特徵計算：根據目標座標計算道路距離、500 公尺環域 POI 數量、核心區判斷與空間滯後特徵。
- 租金預測：結合使用者輸入的房屋屬性與後端空間特徵，代入 LightGBM 模型預測每坪租金。

## 專案資料需求

部署時，repository 需包含下列資料與圖層：

- `Taichung_rental_houses_v4.gpkg`
- `POIs/112Taichung_road_network.gpkg`
- `POIs/Taichung_rail_stations.gpkg`
- `POIs/Taichung_MRT.gpkg`
- `POIs/Taichung_youbikes.gpkg`
- `POIs/Taichung_highway_inters.gpkg`
- `POIs/Taichung_parks.gpkg`
- `POIs/Taichung_schools.gpkg`
- `POIs/Taichung_temples.gpkg`
- `POIs/Taichung_stores.gpkg`
- `POIs/Taichung_busstops.gpkg`
- `POIs/Taichung_medical_service.gpkg`
- `POIs/taichung_town_joined_2.gpkg`

注意：YouBike 區位特徵必須使用 `POIs/Taichung_youbikes.gpkg` 中的 `youbike20` 圖層。本專案不會以其他資料替代 YouBike 圖資。

## 主要檔案與資料夾

- `app.py`：Solara WebApp 入口檔，載入 `src.ui.Page`。
- `src/config.py`：集中管理資料路徑、模型特徵、POI 圖層名稱、核心區定義與特徵說明。
- `src/model.py`：負責 LightGBM 模型訓練、載入、標準化、特徵工程、模型指標與租金預測。
- `src/spatial.py`：負責讀取 POI、道路網、行政區圖層，並計算道路距離、環域數量、直線距離與核心區變數。
- `src/map_view.py`：負責 leafmap / ipyleaflet 互動式地圖、租屋點位圖層與地圖點選座標事件。
- `src/ui.py`：負責 Solara 頁籤、首頁儀表板、地圖預測頁與使用者輸入表單。
- `scripts/train_model.py`：重建 `artifacts/lightgbm_rent_model.joblib`。
- `scripts/build_spatial_cache.py`：重建 `artifacts/spatial_cache.joblib`。
- `artifacts/`：儲存已訓練模型與空間快取，可降低 HuggingFace Space 啟動時間。
- `requirements.txt`：Python 套件需求。
- `Dockerfile`：HuggingFace Docker Space 的容器建置與啟動設定。
- `.gitattributes`：設定 `.gpkg` 與 `.joblib` 為大型二進位檔。
- `.gitignore`：排除快取、暫存與本機環境檔案。

## 本機執行

安裝套件：

```bash
python -m pip install -r requirements.txt
```

可選：預先建立模型與空間快取 artifact：

```bash
python scripts/train_model.py
python scripts/build_spatial_cache.py
```

啟動 Solara：

```bash
python -m solara run app.py --host=127.0.0.1 --port=8765
```

瀏覽器開啟：

```text
http://127.0.0.1:8765
```

## 環境變數

地圖最多渲染點數：

```powershell
$env:APP_MAP_MAX_POINTS="15000"
```

`APP_MAP_MAX_POINTS="0"` 表示不抽樣、顯示全部租屋點。

地圖高度：

```powershell
$env:APP_MAP_HEIGHT="calc(100vh - 96px)"
```

或指定固定高度：

```powershell
$env:APP_MAP_HEIGHT="820px"
```

## HuggingFace Spaces 部署

本專案使用 Docker SDK 部署。HuggingFace Docker Spaces 可支援 Gradio / Streamlit 以外的自訂 WebApp；本專案以 `Dockerfile` 啟動 Solara，並在 `README.md` YAML metadata 中設定：

```yaml
sdk: docker
app_port: 7860
```

詳細部署流程請參考本專案的 `HuggingFace部署說明.txt`。

## 參考文件

- HuggingFace Docker Spaces：https://huggingface.co/docs/hub/main/en/spaces-sdks-docker
- HuggingFace Hub 上傳檔案：https://huggingface.co/docs/huggingface_hub/guides/upload
- HuggingFace Repository 入門：https://huggingface.co/docs/hub/repositories-getting-started
