from __future__ import annotations

import html

import pandas as pd
import solara
import solara.lab

from .config import FEATURE_DESCRIPTIONS, USER_BINARY, USER_CONTINUOUS
from .map_view import MAP_CENTER, create_leafmap_widget
from .model import feature_description_table, load_or_train_bundle, predict_rent_per_ping
from .spatial import compute_location_features, load_or_build_spatial_cache

target_lat = solara.reactive(float(MAP_CENTER[0]))
target_lon = solara.reactive(float(MAP_CENTER[1]))
selected_tab = solara.reactive(0)

BINARY_LABELS = {
    "pet_friendly": "可養寵物",
    "limited": "租屋限制",
    "parking": "停車位",
    "apartment": "公寓",
    "elevator_building": "電梯大樓",
    "air_conditioner": "冷氣",
    "laundry": "洗衣設備",
}

CONT_LABELS = {
    "area_pings": "坪數",
    "deposit_months": "押金月數",
    "mgmt_fee": "管理費",
    "water_fee": "水費",
    "sum_equip_idx": "設備指標總和",
}

APP_CSS = """
:root {
  --ds-ink: #16324f;
  --ds-ink-soft: #385169;
  --ds-green: #0f766e;
  --ds-green-dark: #14532d;
  --ds-rent: #b91c1c;
  --ds-rent-soft: #fee2e2;
  --ds-paper: #f7faf8;
  --ds-line: #dbe4df;
  --ds-line-strong: #b7c8c0;
}
.codex-app-page {
  width: 100%;
  box-sizing: border-box;
  padding: 22px 28px 30px 28px;
  background: linear-gradient(180deg, #f7faf8 0%, #ffffff 58%);
}
.app-panel {
  border: 1px solid var(--ds-line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 10px 28px rgba(20, 83, 45, 0.06);
  box-sizing: border-box;
  width: 100%;
  overflow: hidden;
}
.panel-body {
  padding: 16px 18px;
}
.panel-kicker {
  color: var(--ds-green);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  margin-bottom: 6px;
}
.panel-title {
  color: var(--ds-ink);
  font-size: 22px;
  line-height: 1.25;
  font-weight: 800;
  margin: 0 0 10px 0;
}
.panel-title.small {
  font-size: 18px;
  margin-bottom: 6px;
}
.panel-subtitle,
.panel-copy {
  color: var(--ds-ink-soft);
  font-size: 14px;
  line-height: 1.7;
  margin: 0;
}
.home-grid {
  display: grid;
  grid-template-columns: minmax(620px, 1.08fr) minmax(620px, 1fr);
  gap: 18px;
  align-items: stretch;
  width: 100%;
}
.home-stack {
  display: grid;
  gap: 18px;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(118px, 1fr));
  gap: 12px;
  margin-top: 10px;
}
.metric-tile {
  border: 1px solid var(--ds-line);
  border-radius: 8px;
  padding: 13px 14px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbfa 100%);
  min-height: 78px;
}
.metric-label {
  color: #607085;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 5px;
}
.metric-value {
  color: var(--ds-ink);
  font-size: 26px;
  line-height: 1.1;
  font-weight: 850;
}
.metric-tile:first-child .metric-value {
  color: var(--ds-rent);
}
.app-table-wrap {
  overflow: auto;
  padding-right: 4px;
}
.app-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  line-height: 1.45;
}
.app-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #eef6f3;
  color: #123044;
  border-bottom: 1px solid var(--ds-line-strong);
  padding: 8px 9px;
  text-align: left;
  white-space: nowrap;
}
.app-table td {
  border-bottom: 1px solid #edf2ef;
  padding: 7px 9px;
  vertical-align: top;
}
.app-table tbody tr:nth-child(even) {
  background: #f8fbfa;
}
.app-table tbody tr:hover {
  background: #fff1f1;
}
.map-page {
  padding: 14px 18px 18px 18px;
}
.map-shell {
  border: 1px solid var(--ds-line);
  border-radius: 8px;
  overflow: hidden;
  background: #ffffff;
  box-shadow: 0 10px 28px rgba(20, 83, 45, 0.06);
  position: relative;
}
.map-badge {
  position: absolute;
  top: 14px;
  left: 62px;
  z-index: 500;
  max-width: min(420px, calc(100% - 76px));
  border: 1px solid rgba(183, 200, 192, 0.9);
  border-left: 5px solid var(--ds-rent);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 10px 24px rgba(22, 50, 79, 0.12);
  padding: 10px 12px 11px 12px;
  pointer-events: none;
}
.map-badge-title {
  color: var(--ds-ink);
  font-size: 16px;
  font-weight: 850;
  line-height: 1.25;
}
.map-badge-copy {
  color: var(--ds-ink-soft);
  display: block;
  font-size: 12px;
  line-height: 1.45;
  margin-top: 4px;
}
.control-shell {
  border: 1px solid var(--ds-line);
  border-top: 4px solid var(--ds-green);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.97);
  box-shadow: 0 10px 28px rgba(20, 83, 45, 0.06);
  padding: 16px 18px 18px 18px;
  box-sizing: border-box;
}
.control-hero {
  padding-bottom: 12px;
}
.control-hero-title {
  color: var(--ds-ink);
  font-size: 20px;
  line-height: 1.25;
  font-weight: 850;
  margin: 0 0 8px 0;
}
.control-section {
  border-top: 1px solid #edf2ef;
  padding-top: 12px;
  margin-top: 12px;
}
.section-label {
  color: var(--ds-ink);
  font-size: 15px;
  font-weight: 800;
  margin-bottom: 8px;
}
.coordinate-pill {
  color: var(--ds-green-dark);
  background: #ecfdf5;
  border: 1px solid #bbf7d0;
  border-radius: 999px;
  display: inline-block;
  font-size: 13px;
  font-weight: 700;
  padding: 5px 10px;
}
.result-card {
  border: 1px solid #fecaca;
  border-left: 5px solid var(--ds-rent);
  border-radius: 8px;
  background: #fff7f7;
  padding: 12px 14px;
  margin-top: 14px;
}
.result-value {
  color: var(--ds-rent);
  font-size: 26px;
  font-weight: 850;
  line-height: 1.15;
}
@media (max-width: 1360px) {
  .home-grid {
    grid-template-columns: 1fr;
  }
  .metric-grid {
    grid-template-columns: repeat(2, minmax(140px, 1fr));
  }
}
@media (max-width: 1180px) {
  .map-page {
    flex-direction: column !important;
    overflow: visible !important;
  }
  .map-column,
  .control-column {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    max-width: none !important;
    width: 100% !important;
  }
  .control-column {
    max-height: none !important;
    overflow-y: visible !important;
  }
}
"""


def _yes_no(value: str) -> float:
    return 1.0 if value == "是" else 0.0


def _display_table(df: pd.DataFrame, max_rows: int | None = None):
    table = df if max_rows is None else df.head(max_rows)
    solara.display(table.reset_index(drop=True))


def _table_html(df: pd.DataFrame) -> str:
    return df.reset_index(drop=True).to_html(index=False, border=0, classes="app-table", escape=True)


def _panel_html(title: str, body: str, kicker: str = "", subtitle: str = "") -> str:
    kicker_html = f"<div class='panel-kicker'>{html.escape(kicker)}</div>" if kicker else ""
    subtitle_html = f"<p class='panel-subtitle'>{html.escape(subtitle)}</p>" if subtitle else ""
    return (
        "<section class='app-panel'><div class='panel-body'>"
        f"{kicker_html}<h2 class='panel-title small'>{html.escape(title)}</h2>"
        f"{subtitle_html}{body}</div></section>"
    )


def _table_panel_html(title: str, df: pd.DataFrame, height: str, kicker: str = "", subtitle: str = "") -> str:
    table = _table_html(df)
    body = f"<div class='app-table-wrap' style='height:{height}'>{table}</div>"
    return _panel_html(title, body, kicker=kicker, subtitle=subtitle)


def _metrics_panel_html(metrics: dict, trained_rows: int, created_at: str) -> str:
    labels = [("R2", "R2"), ("RMSE", "RMSE"), ("MAE", "MAE"), ("MAPE_pct", "MAPE %")]
    tiles = "".join(
        f"<div class='metric-tile'><div class='metric-label'>{html.escape(label)}</div>"
        f"<div class='metric-value'>{html.escape(str(metrics[key]))}</div></div>"
        for key, label in labels
    )
    body = (
        f"<div class='metric-grid'>{tiles}</div>"
        f"<p class='panel-subtitle' style='margin-top:12px'>訓練樣本數：{trained_rows:,}；模型建立時間：{html.escape(created_at)}</p>"
    )
    return _panel_html("模型表現指標", body, kicker="MODEL PERFORMANCE")


def _home_html(bundle: dict) -> str:
    feature_df = feature_description_table()
    fi = bundle["feature_importance"].copy()
    fi["說明"] = fi["Feature"].map(FEATURE_DESCRIPTIONS)
    shap_df = bundle["shap_importance"].copy()
    shap_df["說明"] = shap_df["Feature"].map(FEATURE_DESCRIPTIONS)
    intro = (
        "<section class='app-panel'><div class='panel-body'>"
        "<div class='panel-kicker'>TAICHUNG RENTAL PREDICTION</div>"
        "<h1 class='panel-title'>臺中市租屋每坪租金預測 WebApp</h1>"
        "<p class='panel-copy'>本系統使用 <code>Taichung_rental_houses_v4.gpkg</code> 的租屋點位與屬性資料，"
        "依據課堂 notebook 的 F3_SPATIAL 特徵集訓練 LightGBM 模型。預測目標為 <code>ln_rent</code>，"
        "介面會將模型輸出還原為每坪租金。墨綠色代表研究儀表板基調，租金紅用於標示模型與租金重點。</p>"
        "</div></section>"
    )
    return (
        "<div class='codex-app-page'>"
        "<div class='home-grid'>"
        f"{intro}"
        f"{_metrics_panel_html(bundle['metrics'], int(bundle.get('trained_rows', 0)), str(bundle.get('created_at', '尚未記錄')))}"
        "</div>"
        "<div class='home-grid' style='margin-top:18px'>"
        f"{_table_panel_html('特徵變數說明', feature_df, '560px', kicker='FEATURE DICTIONARY', subtitle='模型使用的 29 個 F3_SPATIAL 特徵。')}"
        "<div class='home-stack'>"
        f"{_table_panel_html('LightGBM 特徵重要性', fi, '264px', kicker='FEATURE IMPORTANCE', subtitle='依 LightGBM split/gain 重要性排序。')}"
        f"{_table_panel_html('SHAP 平均絕對貢獻度', shap_df, '264px', kicker='MODEL INTERPRETATION', subtitle='以測試樣本計算各特徵對 ln_rent 的平均絕對貢獻。')}"
        "</div>"
        "</div>"
        "</div>"
    )


@solara.component
def HomePage(bundle: dict):
    solara.HTML(tag="div", unsafe_innerHTML=_home_html(bundle))


@solara.component
def MapPanel():
    with solara.Column(classes=["map-shell"], style={"width": "100%"}):
        map_widget = solara.use_memo(lambda: create_leafmap_widget(target_lat, target_lon), [])
        solara.display(map_widget)
        solara.HTML(
            tag="div",
            unsafe_innerHTML=(
                "<div class='map-badge'><div class='panel-kicker'>INTERACTIVE MAP</div>"
                "<div class='map-badge-title'>租屋點位與目標位置</div>"
                f"<span class='map-badge-copy'>目前目標座標：{target_lon.value:.6f}, {target_lat.value:.6f}</span></div>"
            ),
        )


@solara.component
def ControlPanel(bundle: dict, spatial_cache: dict):
    stats = bundle["feature_stats"]
    continuous_state = {
        key: solara.use_reactive(float(stats[key]["median"])) for key in USER_CONTINUOUS
    }
    binary_state = {key: solara.use_reactive("否") for key in USER_BINARY}
    prediction = solara.use_reactive(None)
    error = solara.use_reactive("")

    with solara.Column(classes=["control-shell"], style={"width": "100%"}):
        solara.HTML(
            tag="div",
            unsafe_innerHTML=(
                "<div class='control-hero'><div class='panel-kicker'>MAP PREDICTION</div>"
                "<h2 class='control-hero-title'>設定目標房屋並估算租金</h2>"
                "<p class='panel-copy'>在左側地圖點選目標位置，或拖曳標記調整座標；點選既有租屋點會顯示該物件屬性並同步設為目標座標。填寫屬性後即可計算預測租金。</p>"
                f"<span class='coordinate-pill' style='margin-top:10px'>目標座標：{target_lon.value:.6f}, {target_lat.value:.6f}</span></div>"
            ),
        )

        with solara.Column(classes=["control-section"]):
            solara.HTML(tag="div", unsafe_innerHTML="<div class='section-label'>房屋物件數值屬性</div>")
            for key, state in continuous_state.items():
                item = stats[key]
                solara.SliderFloat(
                    CONT_LABELS[key],
                    value=state,
                    min=float(item["min"]),
                    max=float(item["max"]),
                    step=float(item["step"]),
                )

        with solara.Column(classes=["control-section"]):
            solara.HTML(tag="div", unsafe_innerHTML="<div class='section-label'>房屋物件類別屬性</div>")
            binary_items = list(binary_state.items())
            for i in range(0, len(binary_items), 2):
                with solara.Row(gap="12px", style={"width": "100%", "align-items": "end"}):
                    for key, state in binary_items[i : i + 2]:
                        with solara.Column(style={"flex": "1 1 0", "min-width": "0"}):
                            solara.Select(BINARY_LABELS[key], values=["否", "是"], value=state)

        def run_prediction():
            try:
                location = compute_location_features(target_lon.value, target_lat.value, spatial_cache)
                user_inputs = {key: float(state.value) for key, state in continuous_state.items()}
                user_inputs.update({key: _yes_no(state.value) for key, state in binary_state.items()})
                result = predict_rent_per_ping(
                    bundle,
                    user_inputs,
                    location["features"],
                    float(location["details"]["x3826"]),
                    float(location["details"]["y3826"]),
                )
                prediction.value = {"prediction": result, "location": location}
                error.value = ""
            except Exception as exc:
                prediction.value = None
                error.value = str(exc)

        solara.Button("計算預測租金", on_click=run_prediction, color="primary", outlined=False)

        if error.value:
            solara.Error(error.value)

        if prediction.value:
            pred = prediction.value["prediction"]
            loc = prediction.value["location"]
            solara.HTML(
                tag="div",
                unsafe_innerHTML=(
                    "<div class='result-card'><div class='panel-kicker'>PREDICTED RENT</div>"
                    f"<div class='result-value'>{pred['rent_per_ping']:,.0f} 元 / 坪</div>"
                    f"<p class='panel-subtitle'>模型輸出 ln_rent = {pred['ln_rent']:.4f}</p></div>"
                ),
            )
            loc_df = pd.DataFrame([loc["details"]]).T.reset_index()
            loc_df.columns = ["區位計算項目", "值"]
            _display_table(loc_df)


@solara.component
def PredictionPage(bundle: dict, spatial_cache: dict):
    with solara.Row(
        gap="18px",
        classes=["codex-app-page", "map-page"],
        style={
            "align-items": "stretch",
            "width": "100%",
            "min-height": "calc(100vh - 96px)",
            "overflow": "hidden",
        },
    ):
        with solara.Column(classes=["map-column"], style={"flex": "1 1 auto", "min-width": "640px"}):
            MapPanel()
        with solara.Column(
            classes=["control-column"],
            style={
                "flex": "0 0 560px",
                "min-width": "500px",
                "max-width": "620px",
                "max-height": "calc(100vh - 96px)",
                "overflow-y": "auto",
            },
        ):
            ControlPanel(bundle, spatial_cache)


@solara.component
def Page():
    solara.Title("臺中市租金預測 LightGBM WebApp")
    solara.HTML(tag="style", unsafe_innerHTML=APP_CSS)
    bundle = solara.use_memo(load_or_train_bundle, [])
    spatial_cache = solara.use_memo(load_or_build_spatial_cache, [])
    with solara.lab.Tabs(value=selected_tab, grow=True, color="#0f766e", slider_color="#b91c1c"):
        solara.lab.Tab("首頁")
        solara.lab.Tab("地圖預測")
    if selected_tab.value == 0:
        HomePage(bundle)
    else:
        PredictionPage(bundle, spatial_cache)
