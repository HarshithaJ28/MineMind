## MineMind
MineMind is a prototype tool for visualizing predicted mine-related hazard risk across terrain and planning safer traversal routes that account for both predicted risk and model uncertainty. It combines geospatial plotting, a confidence-aware pathfinding graph, explainability (SHAP), and active learning utilities — all surfaced through a Streamlit UI.
This README documents the tech stack, quickstart steps, frontend usage summary, backend components, data layout, background jobs, and developer notes.

---

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

---

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

---

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

---

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
---

