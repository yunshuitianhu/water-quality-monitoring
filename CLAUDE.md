# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

水质监测溯源助手 (Water Quality Monitoring & Tracing Assistant) — a Streamlit app + MCP server that uses LLM agents (DeepSeek) to analyze ship-based water quality monitoring data for the 雅瑶水道 (Yayao Waterway) in Foshan, Guangdong. Includes a 1D hydrodynamic + water quality river model with dual-engine support (HEC-RAS 7.0 COM / pure Python Preissmann) and pollution source tracing animation.

## Commands

```bash
# Run the Streamlit app
streamlit run app.py

# One-click launcher (Windows)
双击 启动.bat

# Run MCP server (for Claude Code integration)
python -m water_quality_mcp.server

# Run smoke test (validates all core modules)
python test_smoke.py
# Expected: 8/8 pass

# Run pollution trace-back animation builder (standalone, uses default Excel)
python build_animation.py
# Output: river_pollution_animation.html (self-contained, open in browser)

# Run river model demo
python -c "
from river_model.state import RiverModelState
from river_model.tools import tool_river_init, tool_river_hydro_simulation, tool_river_pollution_event
import warnings; warnings.filterwarnings('ignore')
state = RiverModelState()
tool_river_init(state, n_cross_sections=30)
tool_river_hydro_simulation(state, duration_h=6.0, dt_s=120)
print(tool_river_pollution_event(state, pollutant_type='ammonia', load_kg=200))
"

# Test HEC-RAS COM connectivity (optional, Windows only)
python -c "import win32com.client; c=win32com.client.Dispatch('RAS70.HECRASController'); print(c.HECRASVersion)"
```

## Architecture

### Dual entry points — keep tools in sync

| Entry point | File | Tool registration |
|---|---|---|
| Streamlit | `app.py` | OpenAI function-calling schema in `tools` list + dispatch in `execute_tool()` (22 tools: 16 monitoring + 6 river model; `generate_river_map_animation` excluded to avoid overlap with tab3 heatmap animation) |
| MCP server | `water_quality_mcp/src/water_quality_mcp/server.py` | `@mcp.tool(...)` decorators (23 tools: 16 monitoring/analysis + 7 river model) |

### Data flow (upload → analysis → charts → LLM report)

```
load_data()                     # uploaded_file → df + _uploaded_data.xlsx copy
  → reset all session state
  → delete all old chart files
  → save _uploaded_data.xlsx for build_trace_animation

LLM auto-analysis (11 steps):
  1. get_data_summary          → basic stats JSON
  2. find_black_spots          → dual-standard detection + DBSCAN clustering
  3. reverse_geocode           → address lookup for cluster center
  4. get_national_station_data → nearby reference stations
  5. analyze_video             → frame-by-frame turbidity/color assessment
  5.5. cross_validate_video    → Pearson correlation video vs sensors
  6. search_nearby_pollution_sources → Amap POI search
  7. get_satellite_image       → StarCloud tile lookup
  8. generate_time_series_chart → 4-panel matplotlib PNG
  9. generate_trace_map         → interactive folium HTML
  10. get_comprehensive_summary → unified JSON aggregating all above
  11. generate_final_report     → DeepSeek GB3838 trace report
```

### Dual-standard black-odor detection

The system uses two complementary standards in `analysis.py`:

| Standard | Role | Thresholds |
|---|---|---|
| GB3838-2002 V类 | Quantitative: single-factor pollution index Pi = Ci/V类限值 | NH3≤2.0, DO≥2.0, COD≤40 mg/L |
| Nemerow index | Composite score: P = sqrt((P_avg² + P_max²) / 2) | Severity graded by Pi multiplier |
| 《城市黑臭水体整治工作指南》(2015) | Qualitative: black-odor classification | NH3>8 or DO<2.0 → black-odor; NH3>15 or DO<0.2 → severe |

Severity legend (based on Nemerow index vs V类):
- <1.0 → 达V类 (green)
- 1-2 → 轻污染 (orange)
- 2-3 → 污染 (red-orange)
- 3-5 → 重污染 (crimson)
- >5 → 严重污染 (dark red)

### Monitoring & analysis (`water_quality_mcp/`)

Package installable via `pip install -e water_quality_mcp/` (see `pyproject.toml` for deps).

| Module | Purpose |
|---|---|
| `server.py` | MCP server entry point — 16 monitoring tools + 7 river model tools via `@mcp.tool(...)` |
| `analysis.py` | `find_black_spots()` (dual-standard + DBSCAN), `calc_black_score_vectorized()` (Nemerow), `cross_validate_video_with_monitoring()` (video↔sensor Pearson), `build_comprehensive_summary()` (unified JSON), `data_summary()`, severity/POI helpers |
| `loaders.py` | CSV/Excel data loading with base64 decode, column name normalization |
| `geo_tools.py` | WGS-84 ↔ GCJ-02 coordinate conversion, Amap POI search, reverse geocode, satellite tile lookup |
| `charts.py` | `generate_time_series()` (4-panel), `generate_trace_map()` (folium + Top-5 markers), `generate_cross_validation_chart()` (dual-panel video vs sensor), `generate_correlation_heatmap()` |
| `report.py` | DeepSeek API report generation — sends summary to `deepseek-chat` |
| `national_station.py` | National water quality station data lookup |
| `config.py` | API key resolution (from `.mcp.json` or env vars) |
| `state.py` | `WaterQualityState` dataclass (df, black_spots, video_frames, cross_validation_result) |
| `font_utils.py` | Chinese font resolution for matplotlib charts |

### River model (`river_model/`)

**Engine dispatch**: `hydro_engine.py` detects HEC-RAS COM, falls back to pure Python Preissmann on any failure. For unsteady flow, **always use pure Python** (`pure_py_hydro.py`) with tidal boundary conditions — HEC-RAS COM only runs steady flow.

**Numerical core**:
- `pure_py_hydro.py` — Preissmann 4-point weighted implicit scheme (**φ=0.5, θ=0.6**) for 1D Saint-Venant equations. Double-sweep Thomas algorithm with Newton-Raphson iteration. NaN guard: reverts to previous timestep on divergence.
- `wq_engine.py` — Operator-split ADR solver: (1) advection: explicit upwind, (2) diffusion: explicit central difference, (3) reaction: RK4 with QUAL2E-type kinetics (CBOD/NH₃/DO). **Critical**: velocity must NOT use `np.abs()` — bidirectional flow needed for tidal reversal. `dt_day = dt_s / 86400.0` for rate constant units.

**Other modules**:
- `config.py` — `YayaoConfig` dataclass (40+ parameters: geometry, hydraulics, water quality, GB3838 standards, numerical parameters). `newton_tolerance` defaults to `0.001` (1mm, matching HEC-RAS).
- `cross_sections.py` — `CrossSection` dataclass + `generate_yayao_cross_sections()` — trapezoidal sections
- `state.py` — `HydroResult` (t, chainage, water_level, discharge, velocity, area), `WQResult` (t, chainage, ammonia, cbod, nitrate, dissolved_oxygen, phosphate) — **note: no `concentration` attribute, use `.ammonia` or `.cbod` directly**
- `hecras_bridge.py` — HEC-RAS 7.0 COM automation: `.g01` / `.f01` generation, HDF5 extraction
- `visualization.py` — matplotlib static charts + `build_plotly_animation()`
- `tools.py` — 7 tool wrappers returning JSON strings ≤2000 chars
- `geo_mapping.py` — GPS↔chainage bidirectional mapping (Haversine, Gaussian smoothing, `scipy.interpolate.interp1d`)
- `animation.py` — GB3838 5-class color scales, relative concentration-to-color mapping

### Animation builder (`build_animation.py`)

Standalone script (also callable as `build_trace_animation()` from `app.py`). Produces self-contained `river_pollution_animation.html`:

1. Loads monitoring data from Excel → runs `find_black_spots` → identifies worst cluster
2. Searches nearby factories via Amap POI API (requires valid `AMAP_KEY`)
3. Runs pure Python Preissmann (6h) + ADR simulation (24h, dt=15s) at factory chainage
4. Generates Leaflet HTML with `leaflet-heat.js`:
   - Layer 1: Real monitoring data (smart density sampling)
   - Layer 2: Instantaneous heatmap (baseline-subtracted, GB3838 color scale)
   - Layer 3: Factory markers (red=main suspect, orange=nearby)
   - Controls: play/pause, time slider, speed selector, keyboard shortcuts

**Data integration**: In the Streamlit app, `load_data()` saves uploaded data as `_uploaded_data.xlsx`; tab3 passes it to `build_trace_animation(excel_path="_uploaded_data.xlsx")`. If no data has been uploaded, tab3 shows a warning instead of falling back to the default Excel. When run standalone via CLI, falls back to the hardcoded default Excel path.

### Chart captions system (`app.py`)

Every chart auto-generates a data-driven caption displayed below it via `st.caption()`:
- `_build_time_series_caption()` — NH3 peak, DO min, black-spot count/distribution
- `_build_trace_map_caption()` — cluster count, severity level, legend explanation
- `_build_cross_validation_caption()` — Pearson r pairs, black-odor agreement rate
- `_build_correlation_heatmap_caption()` — strongest correlations
- `_build_river_animation_caption()` — simulation duration, peak concentration/time
- `_render_river_chart_caption()` — per-chart captions for 6 river model chart types

Captions are stored in `st.session_state.chart_captions` dict and rendered in tab2/tab3.

### Session state & data switching

On new data upload, `load_data()` performs a full reset:
1. Clears all session state (black_spots, video_frames, cross_validation_result, chart_captions, anim_html, analysis_done, charts_generated, messages, river_state)
2. Deletes all 12 generated chart files (time_series.png, trace_map.html, cross_validation.png, correlation_heatmap.png, river_*.png, river_pollution_animation.html, _uploaded_data.xlsx)
3. Saves new `_uploaded_data.xlsx` for the animation builder

## Known pitfalls

- **`st.iframe` requires absolute file paths**: `st.iframe("trace_map.html")` → 404 because the bare filename is treated as a URL path. Use `os.path.join(os.path.dirname(__file__), "trace_map.html")` for local files. `st.components.v1.html()` was deprecated (removed after 2026-06-01).
- **`generate_yayao_cross_sections` parameter order**: `mannings_n` is 8th parameter, `bottom_width_m` is 5th. Always use keyword args.
- **WQResult has no `concentration` attribute**: use `wq.ammonia`, `wq.cbod`, `wq.dissolved_oxygen` directly. See `_get_wq_conc()` helper in app.py.
- **API key naming**: canonical name is `DEEPSEEK_API_KEY`. Code also reads `DASHSCOPE_API_KEY` for backward compatibility. `.env.example` and setup wizard both write `DEEPSEEK_API_KEY`.
- **HEC-RAS COM hangs**: kill leftover `Ras.exe` in Task Manager if `Project_Open` hangs.
- **Preissmann solver**: 6h/30s is stable; longer durations (>12h) produce NaN. Newton tolerance is 0.001 (1mm), not smaller. NaN guard reverts to previous timestep.
- **HEC-RAS 7.0 geometry format**: cross sections use `Type RM Length L Ch R = 1 ,<station>,<L>,<C>,<R>` line (no `Node Name=`/`River Station=`), Manning n without leading zero (`.033` not `0.033`), fixed-width 8-char columns.
- **WQ engine velocity**: must NOT use `np.abs()` — bidirectional flow needed for tidal reversal.
- **WQ engine time units**: reaction rate constants are in d⁻¹; internal time step must convert via `dt_day = dt_s / 86400.0`.
- **Amap API**: key in `.mcp.json` as `AMAP_KEY`. Search radius 1000m, keywords: 工厂/企业/养殖/化工/工业区. Amap uses GCJ-02 coordinates — `geo_tools.py` handles WGS-84 ↔ GCJ-02 conversion automatically.
- **Coordinate systems**: monitoring data GPS is WGS-84; Amap POI search returns GCJ-02; satellite tiles use GCJ-02. Always convert through `geo_tools.py` functions.
- **`calc_black_score_vectorized` returns numpy array, not pandas Series**: use `scores[i]` not `scores.iloc[i]`.
- **Tool return truncation**: `process_user_message` in app.py truncates tool results at 2000 chars (4000 for `get_comprehensive_summary`). Keep tool return JSONs concise.
- **`RiverModelState` is recreated on data switch**: tab3 model state is lost when uploading new data — user must re-run the river model.
- **Font discovery**: use `font_utils.discover_font()` (cross-platform) rather than hardcoding `C:/Windows/Fonts/simhei.ttf`. `app.py` and `visualization.py` both use this now.
- **`启动.bat` port handling**: auto-tries 8501 → 8502 → 8503 on conflict. Does NOT kill existing processes.

## Environment

- Developer machine: Windows 11, Python 3.14
- Target: Python 3.10 ~ 3.12 (recommend 3.11); cross-platform (Windows/macOS/Linux)
- Install: `pip install -r requirements.txt` (loose) or `pip install -r requirements-lock.txt` (pinned versions)
- HEC-RAS 7.0 (optional, Windows only) — COM: run `_Register_New_RAS_and_RASMapper_Files.bat` as admin
- API keys configured via first-run setup wizard → saved to `.env` (`DEEPSEEK_API_KEY`) and `.mcp.json`
- Streamlit config: max upload 2048 MB (`.streamlit/config.toml`)
- Key pip packages: `streamlit`, `mcp>=1.0`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `opencv-python`, `matplotlib`, `folium`, `openai`, `openpyxl`, `requests`
- Test: `python test_smoke.py` (8 tests: imports, Excel, black-spots, hydro, WQ, charts)
