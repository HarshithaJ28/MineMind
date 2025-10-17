## MineMind
MineMind is a prototype tool for visualizing predicted mine-related hazard risk across terrain and planning safer traversal routes that account for both predicted risk and model uncertainty. It combines geospatial plotting, a confidence-aware pathfinding graph, explainability (SHAP), and active learning utilities — all surfaced through a Streamlit UI.
This README documents the tech stack, quickstart steps, frontend usage summary, backend components, data layout, background jobs, and developer notes.
## Tech Stack
Frontend:
- Streamlit – Interactive UI and visualization
- Folium + streamlit-folium – Leaflet-based geospatial map rendering
- HTML/CSS (embedded) – Custom UI overlays for better user interaction
Backend & ML:
- Python 3.11
- scikit-learn – Baseline risk prediction models
- SHAP – Explainability for risk models
- imbalanced-learn – SMOTE/resampling for class imbalance
- NetworkX – Graph construction and risk-aware pathfinding
- GeoPandas + Shapely – Geospatial feature engineering
- joblib + pickle – Model persistence
- Custom lightweight job queue – Background task processing (SHAP & Active Learning)
## Quick Start (Windows PowerShell)
1. Activate the project virtual environment (if you use the provided `mrvenv`):
```powershell
C:\Users\harsh\desktop\MineMind\mrvenv\Scripts\Activate.ps1
```
2. Install dependencies (if not already installed):
```powershell
pip install -r requirements.txt
```
3. (Optional but recommended) start the background worker in a separate terminal — this ensures heavy compute (SHAP, active-learning queries) don't block the Streamlit UI:
```powershell
python backend\worker.py
```
4. Launch the frontend:
```powershell
mrvenv\Scripts\python.exe -m streamlit run frontend\app.py
```
Open the URL printed by Streamlit (usually http://localhost:8501).
## Frontend Overview
MineMind's interface helps visualize terrain risk and experiment with model predictions:
- Controls: Adjust risk thresholds, weight factors, and graph density.
- Click-to-Select:
  - Click once for Start, click again for End to generate the shortest, lowest-risk route.
  - Labeling Mode lets you mark regions as cleared or dangerous for active learning.
- Save & Compare Paths: Store Path A, modify parameters, and compare Path B.
- Explainability (SHAP): Explore per-node feature impact and global importance.
- Active Learning: Identify uncertain regions, label them, and retrain models.
- RAG Geospatial Q&A: Query expressions like ">0.7 risk and near rivers" and visualize results.
## Background Jobs
Heavy computations (SHAP, retraining, queries) are run via a lightweight filesystem-backed job queue.
To process queued jobs:
```powershell
python backend\worker.py
```
Each job creates logs in `/jobs/<job_id>/log.txt` and metadata in `/jobs/<job_id>/job.json`.
This system is lightweight and designed for local experimentation, not large-scale deployment.
## Explainability (SHAP)
- SHAP artifacts are stored in `/models/` (`shap_values.npy`, `shap_summary.json`).
- Use the Streamlit UI to trigger SHAP computation (enqueued as a background task).
- Visualize feature contributions:
  - Positive SHAP increases risk
  - Negative SHAP reduces risk
For large datasets, limit sample size to avoid performance issues.
## Active Learning
- Implemented in `backend/active_learning.py`.
- Identifies top-N uncertain samples and writes them to `models/active_queries.json`.
- Use the Streamlit UI to label uncertain regions and retrain models.
- Loop: query → label → retrain → re-evaluate.
## Data Layout
- `data/terrain_data.geojson` / `terrain_data.csv` — Base terrain dataset
- `data/terrain_data_with_risk.csv` — Risk + uncertainty model output
- `data/labels_delta.csv` — Appended labels from user interaction
- `data/rag_query_result.geojson` — Query results overlay
## Developer Notes
- If the Folium map appears dark/overlayed, reload or restart Streamlit.
- If pathfinding fails → increase neighbors per node (k) in settings.
- Use the background worker to prevent UI freezes during heavy computation.
- Consider upgrading the queue to Celery + Redis for production-scale use.
## Reproducing Experiments
Example sequence for generating Phase 1 artifacts:
```powershell
# Activate environment
mrvenv\Scripts\Activate.ps1
# Preprocess dataset
python backend\preprocess.py
# Train baseline model
python backend\train_baselines.py
# Generate reports and metrics
python backend\operating_points.py
# Compute SHAP (optional, via worker)
python backend\worker.py
python backend\shap_explain.py --sample-size 100
# Launch frontend
streamlit run frontend\app.py
```
## Future Enhancements
- Integrate satellite imagery layers (Sentinel-2 / Google Earth Engine)
- Deploy a multi-user version using Streamlit Cloud or FastAPI backend
- Add real-time retraining dashboard for rapid labeling loops
- Replace local queue with Redis RQ / Celery for distributed tasks
If you want, I can also:
- Move Save/Compare controls from the (sometimes hidden) sidebar into the left Controls card.
- Add a visible "Clear start/end" button to the controls.
- Replace the `st_folium` embedding with `components.html` + folium HTML (removes overlay issues permanently).
Questions or changes you'd like prioritized? Open an issue or tell me here and I'll implement it.
## MineRiskMapper

MineRiskMapper is a prototype tool for visualizing predicted mine-related hazard risk across terrain and planning safer traversal routes that account for both predicted risk and model uncertainty. It combines geospatial plotting, a confidence-aware pathfinding graph, explainability (SHAP), and active learning utilities — all surfaced through a Streamlit UI.

This README documents the tech stack, architecture, quickstart steps, frontend usage summary, backend components, data layout, background jobs, and developer notes.

---

## Tech stack

- Python 3.11
- Streamlit for the interactive frontend (`frontend/app.py`)
- Folium + streamlit_folium to embed Leaflet maps
- GeoPandas / Shapely for geospatial data processing
- scikit-learn for baseline models and graph-based path weighting
- SHAP for model explainability (artifact generation is offloaded to a background worker)
- imbalanced-learn (optional) for SMOTE/resampling experiments
- NetworkX for constructing the nearest-neighbour routing graph
- A tiny filesystem-backed job queue + worker for background tasks (under `backend/`)

---

## Project layout (important files)

- `frontend/app.py` — Streamlit app (main UI, controls, map embedding, click-to-label, A/B path comparison).
- `backend/` — backend scripts & helpers including:
  - `job_queue.py`, `worker.py` — simple filesystem job queue and worker
  - `shap_explain.py` — SHAP computation helper (meant to be run as a background job)
  - `active_learning.py` — query & retrain helpers for active learning flows
  - `preprocess.py`, `train_baselines.py`, `train_calibrated.py` (where present) — data preparation and modeling scripts
- `data/` — source and generated geo files (e.g. `terrain_data.geojson`, `terrain_data.csv`, `labels_delta.csv`, `rag_query_result.geojson`)
- `models/` — saved trained models and SHAP artifacts (e.g. `risk_model.pkl`, `shap_values.npy`, `shap_summary.json`)
- `reports/` — evaluation plots and operating points (ROC/PR/Calibration images, `metrics.json`)
- `UsageGuide.md` — a human-oriented walk-through for the frontend (created alongside this README)
- `requirements.txt` — Python dependencies

---

## Quick start (Windows PowerShell)

1. Activate the project virtual environment (if you use the provided `mrvenv`):

```powershell
C:\Users\harsh\desktop\MineRiskMapper\mrvenv\Scripts\Activate.ps1
```

2. Install dependencies (if not already installed):

```powershell
pip install -r requirements.txt
```

3. (Optional but recommended) start the background worker in a separate terminal — this ensures heavy compute (SHAP, active-learning queries) don't block the Streamlit UI:

```powershell
python backend\worker.py
```

4. Launch the frontend:

```powershell
mrvenv\Scripts\python.exe -m streamlit run frontend\app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

---

## Frontend usage summary

For a detailed step-by-step, see `UsageGuide.md`. Key interactions:

- Controls (left card): risk threshold, risk/uncertainty weights, neighbors per node (k), SHAP toggles, active-learning triggers, and label mode.
- Click-to-select: with `Label mode = None`, the first click selects Start, the second click selects End. The app snaps clicks to the nearest indexed node and computes a weighted shortest path.
- Labeling mode: when `Mark cleared (0)` or `Mark dangerous (1)` is selected, clicks append a label into `data/labels_delta.csv` instead of selecting start/end.
- Save/Compare: save a path as A, then compute a comparison path B with alternate weights to compare cost, risk and length.
- SHAP: show SHAP summaries (when computed), inspect per-node SHAP contributions for clicked nodes. SHAP computation is expensive and should be enqueued as a background job.
- Active Learning: enqueue a backend query to identify top‑N uncertain samples, label them, and retrain using the `active_learning` helper.
- RAG (geospatial Q&A): a small deterministic parser for constrained geospatial queries (e.g. "> 0.7 risk and near rivers"). Results may be saved to `data/rag_query_result.geojson` and can be overlaid on the map.

---

## Background jobs & worker

The UI enqueues heavy jobs (SHAP compute, active-learning queries) into a simple filesystem-backed queue (see `backend/job_queue.py`). To process jobs start the worker in another terminal:

```powershell
python backend\worker.py
```

The worker picks jobs, runs the command, and writes logs under `jobs/<job_id>/log.txt` and metadata in `jobs/<job_id>/job.json`.

Note: the worker is intentionally lightweight and synchronous — it's suitable for local experimentation, not production-scale orchestration.

---

## SHAP explainability

- SHAP artifacts (feature list, shap_values) are stored under `models/`. Use the UI to request a SHAP compute job; the worker will create `models/shap_values.npy` and `models/shap_summary.json`.
- SHAP shows per-node additive contributions: features with positive SHAP increase predicted risk, negative SHAP lower it. Use the mean |SHAP| summary to see global feature importance.
- SHAP compute can be memory- and time-intensive; prefer running it via the worker with a sample-size argument when dataset is large.

---

## Active learning

- Active learning in `backend/active_learning.py` provides a `query` routine that writes top‑N uncertain samples to `models/active_queries.json` and a `retrain_with_labels()` routine that reads `data/labels_delta.csv` to update the labeled set and retrain.
- Workflow: enqueue query job -> review queries in the UI -> label uncertain nodes -> retrain using the helper -> re-evaluate the model.

---

## Data layout

- `data/terrain_data.geojson` / `data/terrain_data.csv` — point dataset with features and geometry
- `data/terrain_data_with_risk.csv` — dataset exported by the app with `risk` and `uncertainty` columns
- `data/labels_delta.csv` — appended labels from the UI (index,label)
- `data/rag_query_result.geojson` — geojson saved by RAG queries when applicable

---

## Developer notes & troubleshooting

- Map overlay / dark card overlay: the UI uses a small amount of inline HTML/CSS to style left/right cards. If you see a semi-transparent dark overlay on top of the Folium map, fully reload the Streamlit page or restart Streamlit. The robust fix is to render the Folium HTML directly inside the card using `streamlit.components.v1.html` (I can apply this change if you'd like).
- If clicks don't snap to the expected node, zoom in and click close to a visible marker — the nearest-node search is computed on Web-Mercator (EPSG:3857) coordinates.
- If path search fails (NetworkX raises no-path), increase `Neighbors per node (k)` and try again to make the graph denser.
- Use the background worker to process expensive jobs — otherwise Streamlit will block while running them.

---

## Running and reproducing experiments

Typical steps used to reproduce Phase1 artifacts (example):

```powershell
# Activate env
mrvenv\Scripts\Activate.ps1

# preprocess
python backend\preprocess.py

# train baselines & produce evaluation reports
python backend\train_baselines.py

# generate operating points
python backend\operating_points.py

# (optional) compute SHAP via the worker
python backend\worker.py  # in separate terminal
# then enqueue SHAP from the UI or run:
python backend\shap_explain.py --sample-size 100

# run the UI
streamlit run frontend\app.py
```

---

## Contributing and extending

- Replace the example dataset in `data/` with higher-fidelity terrain and mine-event datasets for better results. Observe data privacy and safety constraints.
- Consider switching the job queue to a proper task queue (Redis + RQ / Celery) for multi-machine workloads.
- Add unit tests around `backend/active_learning.py` and `backend/shap_explain.py` and CI checks.

---

If you want, I can also:
- Move Save/Compare controls from the (sometimes hidden) sidebar into the left Controls card.
- Add a visible "Clear start/end" button to the controls.
- Replace the `st_folium` embedding with `components.html` + folium HTML (removes overlay issues permanently).

Questions or changes you'd like prioritized? Open an issue or tell me here and I'll implement it.
