# MineRiskMapper — Frontend Usage Guide

This file explains how to use the Streamlit frontend (interactive map, path planning, labeling, SHAP explainability, Active Learning and the lightweight RAG geospatial query helper). Keep this next to the project root so field users and reviewers can find instructions quickly.

---

## Quick start

1. Activate the virtual environment (Windows PowerShell):

```powershell
C:\Users\harsh\desktop\MineRiskMapper\mrvenv\Scripts\Activate.ps1
```

2. Start the background worker (recommended for heavy tasks such as SHAP and active-learning queries):

```powershell
python backend\worker.py
```

3. Launch the Streamlit frontend:

```powershell
mrvenv\Scripts\python.exe -m streamlit run frontend\app.py
```

4. Open your browser at the address Streamlit prints (usually http://localhost:8501).

---

## Layout overview

The frontend has two main columns:
- Left controls card — pathfinding parameters, explainability / active learning toggles, labeling controls and small helper buttons.
- Right (main) card — app title, RAG text input (geospatial Q&A), the interactive map, and evaluation/report panels.

There is also a compact Background Jobs panel accessible from the Streamlit sidebar (used to view logs and enqueued jobs).

---

## Controls (left card)

- Risk Threshold: a slider controlling the threshold for some operating-point computations. Default: 0.5.
- Risk weight in path cost: influence of predicted risk on path cost. Larger values make the algorithm avoid higher-risk nodes more strongly.
- Uncertainty weight in path cost: influence of model uncertainty on path cost. Larger values prefer lower-uncertainty routes.
- Neighbors per node (k): how many nearest neighbors are used to build the routing graph. Increase to make the graph denser (and possibly more connected), decrease to make routes more local.
- Use calibrated probabilities (checkbox): when available, use a calibrated model's predict_proba output. If none is present the app will fall back to the baseline model.
- Retrain model (button): retrain (or load) the model using the current dataset + labels. Use after adding labels.

Labeling / selection
- Label mode for clicked cell (radio):
  - None — clicking selects Start then End nodes (first click becomes Start, second becomes End). Use this to create a path.
  - Mark cleared (0) — clicking will append a label "0" (cleared) for the nearest node into `data/labels_delta.csv`.
  - Mark dangerous (1) — clicking will append a label "1" (dangerous) for the nearest node into `data/labels_delta.csv`.

Notes about clicking: the app uses a KDTree on the metric-projected coordinates to snap each click to the nearest indexed node. Clicks that are far from any node may still snap to a remote node: zoom in to select precisely.

---

## Map interactions

- Pan and zoom using the map controls or mouse gestures.
- Click once near a marker to select Start (when Label mode = None). A UI confirmation appears.
- Click again near a second marker to select End. The app computes a shortest path in the graph weighted by distance + risk + uncertainty (according to the weights you set).
- The selected path is drawn on the map (blue polyline by default). Start and End are shown with markers.
- To label a cell instead of selecting start/end, set the Label mode to "Mark cleared (0)" or "Mark dangerous (1)" and click the node. Labels are appended to `data/labels_delta.csv`.

Saving and comparing paths
- Save the current path as Path A using the left control (or the Sidebar button if present). This stores path A in session-state so you can compare later.
- To compute a comparison Path B, set alternate weights and press the compare button. Path B will be computed and both A and B are displayed (A = blue, B = magenta).

---

## Explainability (SHAP)

- Show SHAP explanations (checkbox): when toggled, if SHAP artifacts are already computed the UI shows a small SHAP summary chart and allows inspecting per-node contributions.
- SHAP sample size: number of samples to use when recomputing SHAP in the backend. Use 0 or empty for the full dataset (may be very slow).
- (Re)compute SHAP in backend (button): enqueues a background SHAP compute job (requires the backend worker to be running to avoid blocking Streamlit). The worker writes artifacts in `models/` such as `shap_values.npy` and `shap_summary.json`.

What SHAP tells you
- SHAP (SHapley Additive exPlanations) decomposes the model's prediction for a specific node into additive contributions from each feature.
- For a clicked node (when SHAP data is available) the app lists top contributing features and their importances (positive contributions push predicted risk higher, negative contributions push it lower).
- Typical usage:
  - Use SHAP to see why the model flagged a node as high risk (which features contributed most).
  - Use the mean |SHAP| summary to identify features with the largest global influence on the model.

Caveats
- SHAP compute can be slow and memory-heavy. Use the background worker or run `python backend/shap_explain.py --sample-size 100` on the machine to compute artifacts offline.

---

## Active Learning

- "Show active-learning queries" displays any query file produced by `backend/active_learning.py` (saved as `models/active_queries.json`).
- "Query uncertain samples (backend)" enqueues a background job that writes the top-N uncertain samples into `models/active_queries.json` for review. The worker must be running to process the job.
- From the displayed queries you can quickly label an index and then click "Retrain model from labels" to run a retrain that uses the appended labels (calls `backend.active_learning.retrain_with_labels()`).

Workflow suggestion
1. Run an uncertain-query job.
2. Inspect the saved queries and label the most suspicious ones using the quick label controls.
3. Retrain with labels and re-evaluate the model.

---

## Background jobs panel

- Opens in the Streamlit sidebar. Lists enqueued jobs and lets you view logs for each job.
- Enqueue a test job from the UI to ensure the worker is connected.
- Start the worker in a separate terminal using `python backend/worker.py`.

---

## Geospatial Q&A (RAG)

- The small RAG helper accepts constrained natural-language queries (examples: `">0.7 risk and near rivers"`, `"high risk within 10 km of event"`).
- Use the RAG box in the main header area. Press "Run RAG query" to execute the local retrieval/parsing helper; if results include a geojson path the app will attempt to display it on the map.
- The RAG is intentionally lightweight and deterministic — it recognizes a few common patterns (thresholds, proximity filters) and saves a geojson (if applicable) into `data/`.

---

## Troubleshooting & tips

- Map dark overlay: if you see a semi-transparent dark layer over the map, fully reload the Streamlit page (Ctrl+Shift+R) — or restart Streamlit — to pick up recent layout changes. The overlay can occur if raw HTML wrappers and Streamlit component rendering get stacked incorrectly. If the overlay persists I recommend switching to the `components.html` approach for the map (ask me to apply this change).
- Clicks not recognizing nodes: zoom in and ensure you click close to a green marker; the nearest-node snapping is computed on the metric-projected coordinates (EPSG:3857).
- Path search fails: graph may be disconnected. Increase `Neighbors per node (k)` and try again.
- SHAP jobs take long: run them via the background worker or use a smaller `--sample-size`.

---

## Developer notes

- Main frontend: `frontend/app.py` — contains control wiring and map rendering. The app loads `data/terrain_data.geojson` (or CSV fallback) and writes label deltas to `data/labels_delta.csv`.
- Background jobs: `backend/job_queue.py` and `backend/worker.py` manage filesystem-enqueued jobs and logs.
- SHAP helper: `backend/shap_explain.py` — produces `models/shap_values.npy` and `models/shap_summary.json`.
- Active learning: `backend/active_learning.py` — query and retrain utilities.

---

If you'd like, I can:
- Move the Save/Compare controls into the left Controls card (so users don't need to open the sidebar).
- Add a "Clear start/end" button to reset selection.
- Convert map rendering to `components.html` to remove overlay issues permanently.

Tell me which of the above you'd like and I'll apply the change.
