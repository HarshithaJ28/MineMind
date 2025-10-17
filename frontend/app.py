# frontend/app.py (Phase 1: Hazard Prediction + Heatmap)

import streamlit as st
import os
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree
import networkx as nx
import numpy as np
import pickle
from shapely.geometry import Point
import branca.colormap as cm
import joblib
import importlib
import sys
import os
# Ensure project root is on sys.path so `import backend.*` works when Streamlit runs
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Import backend.job_queue; if package import fails (Streamlit environment), load from file path
try:
    import backend.job_queue as job_queue
except Exception:
    try:
        import importlib.util
        bj_path = os.path.join(ROOT, 'backend', 'job_queue.py')
        spec = importlib.util.spec_from_file_location('backend.job_queue', bj_path)
        job_queue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job_queue)
    except Exception as e:
        # fallback: mark job_queue unavailable but do not crash the app here.
        job_queue = None
        import traceback
        tb = traceback.format_exc()
        try:
            # st is available (imported earlier); show a UI message when possible
            st.warning('backend.job_queue could not be imported; background jobs disabled. See server logs for details.')
        except Exception:
            pass
        print('backend.job_queue import fallback failed:', tb)


@st.cache_data
def load_gdf():
    """Load terrain GeoDataFrame from geojson or CSV fallback."""
    try:
        gdf = gpd.read_file('data/terrain_data.geojson')
    except Exception:
        df = pd.read_csv('data/terrain_data.csv')
        if 'geometry' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df.geometry), crs='EPSG:4326')
        else:
            gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs='EPSG:4326')
    # Ensure lat/lon columns for convenience
    if 'lat' not in gdf.columns or 'lon' not in gdf.columns:
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
    return gdf.reset_index(drop=True)


def train_model(gdf, features, model_path='models/risk_model.pkl'):
    """Train a RandomForest model (or load if exists)."""
    X = pd.get_dummies(gdf[features + ['land_use']])
    y = gdf['label']
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception:
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X, y)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    return model


def predict_with_uncertainty(model, gdf, features):
    """Return risk (prob of positive) and uncertainty (std of tree votes).

    Uncertainty approx: std of individual tree probabilities (per-tree predict_proba)
    """
    X = pd.get_dummies(gdf[features + ['land_use']])
    # align columns in case of missing categories
    model_features = getattr(model, 'feature_names_in_', None)
    if model_features is not None:
        for c in model_features:
            if c not in X.columns:
                X[c] = 0
        X = X[model_features]

    # Keep X as a pandas DataFrame and pass it directly to predict_proba.
    # This avoids the sklearn UserWarning about invalid feature names when the model
    # was fitted with feature names.
    # Decide whether to call predict_proba with DataFrame or numpy array
    try:
        if hasattr(model, 'feature_names_in_'):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict_proba(X.to_numpy())[:, 1]
    except Exception:
        # fallback to numpy
        probs = model.predict_proba(X.to_numpy())[:, 1]

    # Per-tree probabilities: some tree estimators were sometimes trained from numpy (no feature names)
    try:
        estimators = getattr(model, 'estimators_', [])
        # Always call tree estimators with numpy arrays to avoid warnings when trees
        # were trained without feature names (many sklearn trees don't store them).
        X_np = X.to_numpy()
        all_tree_probs = np.stack([est.predict_proba(X_np)[:, 1] for est in estimators], axis=1)
        uncertainty = all_tree_probs.std(axis=1)
    except Exception:
        # Fallback: use entropy-based uncertainty on probs
        uncertainty = - (probs * np.log(probs + 1e-9) + (1 - probs) * np.log(1 - probs + 1e-9))
        uncertainty = uncertainty / (uncertainty.max() + 1e-9)
    except Exception:
        # Fallback: use entropy-based uncertainty on probs
        uncertainty = - (probs * np.log(probs + 1e-9) + (1 - probs) * np.log(1 - probs + 1e-9))
        # normalize entropy to [0,1]
        uncertainty = uncertainty / (uncertainty.max() + 1e-9)

    return probs, uncertainty


def load_shap_artifacts(shap_dir='models'):
    """Load precomputed SHAP artifacts if available."""
    import json
    import numpy as _np
    feats = None
    shap_vals = None
    top_csv = None
    f_feat = os.path.join(shap_dir, 'shap_features.json')
    f_vals = os.path.join(shap_dir, 'shap_values.npy')
    f_top = os.path.join(shap_dir, 'shap_top_contribs.csv')
    if os.path.exists(f_feat) and os.path.exists(f_vals):
        with open(f_feat, 'r') as f:
            feats = json.load(f)
        shap_vals = _np.load(f_vals)
    if os.path.exists(f_top):
        top_csv = f_top
    return feats, shap_vals, top_csv


def build_graph(gdf, risk, uncertainty, k=8, risk_weight=5.0, uncert_weight=5.0):
    """Build a networkx graph from points. Edge weight combines distance (meters), risk and uncertainty.

    Returns graph and KDTree on metric coords.
    """
    # Project to metric CRS for distance (Web mercator)
    metric = gdf.to_crs(epsg=3857)
    coords = np.vstack([metric.geometry.x.values, metric.geometry.y.values]).T
    tree = KDTree(coords)
    G = nx.Graph()
    n = len(gdf)
    # normalize risk/uncertainty
    r = (risk - np.min(risk)) / (np.ptp(risk) + 1e-9)
    u = (uncertainty - np.min(uncertainty)) / (np.ptp(uncertainty) + 1e-9)

    for i in range(n):
        G.add_node(i, pos=(gdf.iloc[i].lat, gdf.iloc[i].lon), risk=float(r[i]), uncert=float(u[i]))

    distances, indices = tree.query(coords, k=k + 1)
    for i in range(n):
        for j_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
            j = int(j_idx)
            # weight: distance (meters) scaled + risk & uncertainty preference
            weight = (dist / 1000.0) + risk_weight * (r[i] + r[j]) / 2.0 + uncert_weight * (u[i] + u[j]) / 2.0
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(weight), distance=float(dist))

    return G, tree


def nearest_node(tree, gdf, click_lon, click_lat):
    """Find nearest node index to clicked lon/lat.

    Tree expects metric coords (EPSG:3857)
    """
    metric = gdf.to_crs(epsg=3857)
    pt = gpd.GeoSeries([Point(click_lon, click_lat)], crs='EPSG:4326').to_crs(epsg=3857)
    coord = np.array([[pt.geometry.x.iloc[0], pt.geometry.y.iloc[0]]])
    dist, idx = tree.query(coord, k=1)
    return int(idx[0][0])


def draw_map(gdf, risk, uncertainty, path_idx=None, start_idx=None, end_idx=None, risk_threshold=0.5):
    center = [gdf.lat.mean(), gdf.lon.mean()]
    m = folium.Map(location=center, zoom_start=11)
    # color map
    colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=1).to_step(10)
    colormap.caption = 'Predicted risk'
    colormap.add_to(m)

    for i, row in gdf.iterrows():
        clr = colormap(risk[i])
        radius = 4 + 6 * float(uncertainty[i])
        folium.CircleMarker(location=[row.lat, row.lon], radius=radius, color=clr, fill=True, fill_color=clr, fill_opacity=0.7,
                            popup=folium.Popup(f"Index: {i}<br>Risk: {risk[i]:.3f}<br>Uncertainty: {uncertainty[i]:.3f}", max_width=250)).add_to(m)

    # draw path if exists
    if path_idx is not None and len(path_idx) > 1:
        path_coords = [[gdf.iloc[i].lat, gdf.iloc[i].lon] for i in path_idx]
        folium.PolyLine(path_coords, color='blue', weight=4, opacity=0.9).add_to(m)
        # mark start/end
        folium.CircleMarker(location=path_coords[0], radius=8, color='blue', fill=True, fill_color='blue', popup='Start').add_to(m)
        folium.CircleMarker(location=path_coords[-1], radius=8, color='black', fill=True, fill_color='black', popup='End').add_to(m)
    else:
        if start_idx is not None:
            folium.Marker([gdf.iloc[start_idx].lat, gdf.iloc[start_idx].lon], popup='Start', icon=folium.Icon(color='green')).add_to(m)
        if end_idx is not None:
            folium.Marker([gdf.iloc[end_idx].lat, gdf.iloc[end_idx].lon], popup='End', icon=folium.Icon(color='red')).add_to(m)

    return m


def save_with_risk(gdf, risk, uncertainty, outpath='data/terrain_data_with_risk.csv'):
    df = gdf.copy()
    df['risk'] = risk
    df['uncertainty'] = uncertainty
    df.to_csv(outpath, index=False)


### App UI
st.set_page_config(layout='wide', page_title='MineRiskMapper')
st.title('MineRiskMapper — Confidence-Aware Safe Path')
st.markdown('Interactive hazard map with click-to-label and confidence-aware pathfinding')

 

# UI styling: hide default Streamlit sidebar and add dark card styles to emulate the design
st.markdown("""
<style>
/* hide default sidebar area */
section[data-testid="stSidebar"] {display: none;}
/* page background and text */
div[data-testid="stAppViewContainer"] {background: #0f1720; color: #e6eef6}
/* card style for left controls */
.left-card {background:#0b1220; padding:18px; border-radius:12px; box-shadow: 0 8px 30px rgba(0,0,0,0.6);}
.card-title {color:#ffffff; font-weight:700; font-size:20px; margin-bottom:8px}
.muted {color:#9aa6b2}
/* value badge to the right of sliders */
.val-badge {background:#212831; padding:6px 10px; border-radius:10px; color:#e6eef6; text-align:center; display:inline-block}
/* main card style */
.main-card {background: linear-gradient(180deg,#121417,#191b1e); padding:20px; border-radius:12px; box-shadow:0 10px 40px rgba(0,0,0,0.6);}
/* small grid icon */
.grid-icon {width:28px; height:28px; background:rgba(255,255,255,0.03); border-radius:6px; display:inline-block}
/* primary button style (Run RAG) */
button.stButton, .stButton>button {background: linear-gradient(180deg,#2b83ef,#2b6fdc) !important; color: #fff !important; border-radius:8px !important; padding:8px 14px !important}
/* style the folium iframe/map holder for rounded corners */
.stFrame iframe {border-radius:10px !important; box-shadow: 0 6px 20px rgba(0,0,0,0.6) !important;}
/* tweak text widgets */
div[data-baseweb="stMarkdown"] p {color:#e6eef6}
</style>
""", unsafe_allow_html=True)

# Hide the original sidebar widgets by not executing them when HIDE_SIDEBAR=True
HIDE_SIDEBAR = True

# Left column controls recreated as a styled card
def slider_with_badge(label, minv, maxv, val, step, key, fmt='{:.2f}'):
    c0, c1 = st.columns([4, 1])
    with c0:
        v = st.slider(label, minv, maxv, val, step=step, key=key)
    with c1:
        st.markdown(f"<div class='val-badge'>{fmt.format(v)}</div>", unsafe_allow_html=True)
    return v

left_col, main_col = st.columns([1.0, 3.5])
with left_col:
    st.markdown('<div class="left-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Pathfinding Parameters</div>', unsafe_allow_html=True)
    risk_threshold = slider_with_badge('Risk Threshold', 0.0, 1.0, 0.5, 0.01, 'left_risk_threshold')
    risk_weight = slider_with_badge('Risk weight in path cost', 0.0, 20.0, 5.0, 0.5, 'left_risk_weight')
    uncert_weight = slider_with_badge('Uncertainty weight in path cost', 0.0, 20.0, 5.0, 0.5, 'left_uncert_weight')
    k_nn = slider_with_badge('Neighbors per node (k)', 4, 16, 8, 1, 'left_k_nn', fmt='{:.0f}')
    use_calibrated = st.checkbox('Use calibrated probabilities (if available)', value=True, key='left_use_calibrated')
    retrain = st.button('Retrain model (use after labeling)', key='left_retrain')
    st.markdown('---')
    st.write('Click on map to select start/end or to label a cell (use the selection below)')
    label_mode = st.radio('Label mode for clicked cell', ('None', 'Mark cleared (0)', 'Mark dangerous (1)'), key='left_label_mode')
    st.markdown('---')
    st.markdown('### Explainability & Active Learning')
    show_shap = st.checkbox('Show SHAP explanations (precomputed)', value=False, key='left_show_shap')
    shap_sample_size = st.number_input('SHAP sample size (None=full)', min_value=0, value=0, step=50, key='left_shap_sample_size')
    run_shap_compute = st.button('(Re)compute SHAP in backend', key='left_run_shap')
    st.markdown('Active Learning: review uncertain nodes and label them')
    al_show_queries = st.checkbox('Show active-learning queries', value=True, key='left_al_show_queries')
    al_query_n = st.number_input('Query top-n uncertain', min_value=1, max_value=200, value=20, key='left_al_query_n')
    al_run_query = st.button('Query uncertain samples (backend)', key='left_al_run_query')
    al_use_partial = st.checkbox('Prefer incremental retrain (SGD partial_fit) for large data', value=True, key='left_al_use_partial')
    st.markdown('---')
    st.markdown('<div class="muted">Dataset saved to <code>data/terrain_data_with_risk.csv</code></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with main_col:
    # Main card header (title, subtitle, small grid icon)
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    hdr_l, hdr_r = st.columns([9,1])
    with hdr_l:
        st.markdown('<h2 style="margin:0; color:#ffffff">MineRiskMapper — <span style="font-weight:400">Confidence-Aware Safe Path</span></h2>', unsafe_allow_html=True)
    with hdr_r:
        st.markdown('<div class="grid-icon"></div>', unsafe_allow_html=True)
        if 'show_help' not in st.session_state:
            st.session_state.show_help = False
        if st.button('Help', key='top_help'):
            st.session_state.show_help = not st.session_state.show_help
    if st.session_state.show_help:
        with st.expander('Quick tour & help', expanded=True):
            st.markdown('Follow these steps to explore the app:')
            st.markdown('1. Inspect the map: color = predicted risk, radius = model uncertainty.')
            st.markdown('2. Click to select start and end points (or switch to Label mode to add labels).')
            st.markdown('3. Save a path as A, then adjust weights and compute a comparison path B.')
            st.markdown('4. Use SHAP (precomputed) to understand feature contributions; enqueue SHAP compute for more data.')
            st.markdown('5. Use Active Learning to generate uncertain queries, label them, and retrain.')
            st.markdown('6. View Background Jobs to see logs for long-running tasks (start `python backend/worker.py`).')
    st.markdown('<hr style="border:none; height:1px; background:rgba(255,255,255,0.03); margin:12px 0">', unsafe_allow_html=True)
    # RAG input inside main card
    st.markdown('<div style="margin-bottom:12px"><strong style="color:#ffffff">Geospatial Q&A (RAG)</strong></div>', unsafe_allow_html=True)
    rag_q = st.text_input('', placeholder='Ask a geospatial question (e.g. > 0.7 risk and near rivers)', key='rag_q_main')
    if st.button('Run RAG query', key='run_rag_main2'):
        try:
            import backend.rag as rag
            res = rag.answer_query(st.session_state.get('rag_q_main', ''))
            st.write(res['text'])
            if res.get('path') and os.path.exists(res['path']):
                try:
                    qg = gpd.read_file(res['path'])
                    m2 = draw_map(qg, qg.get('risk', [0]*len(qg)), [0]*len(qg))
                    st_folium(m2, width=700, height=420)
                except Exception:
                    st.write('Result saved to', res.get('path'))
        except Exception as e:
            st.error('RAG failed: ' + str(e))
    st.markdown('<div style="margin-top:12px; color:#ffffff; font-weight:700">Map</div>', unsafe_allow_html=True)
    # create a container in the main column for the map so it stays inside the main card
    map_container = main_col.empty()

gdf = load_gdf()
features = ['dist_to_hist_mine', 'elevation', 'rainfall']

if False:  # original sidebar hidden (moved controls to left card)
    st.header('Controls')
    risk_threshold = st.slider('Risk threshold', 0.0, 1.0, 0.5)
    risk_weight = st.slider('Risk weight in path cost', 0.0, 20.0, 5.0)
    uncert_weight = st.slider('Uncertainty weight in path cost', 0.0, 20.0, 5.0)
    k_nn = st.slider('Neighbors per node (k)', 4, 16, 8)
    use_calibrated = st.checkbox('Use calibrated probabilities (if available)', value=True)
    retrain = st.button('Retrain model (use after labeling)')
    st.markdown('---')
    st.write('Click on map to select start/end or to label a cell (use the selection below)')
    label_mode = st.radio('Label mode for clicked cell', ('None', 'Mark cleared (0)', 'Mark dangerous (1)'))
    st.markdown('---')
    st.header('Explainability & Active Learning')
    show_shap = st.checkbox('Show SHAP explanations (precomputed)', value=False)
    shap_sample_size = st.number_input('SHAP sample size (None=full)', min_value=0, value=0, step=50)
    run_shap_compute = st.button('(Re)compute SHAP in backend')
    st.markdown('Active Learning: review uncertain nodes and label them')
    al_show_queries = st.checkbox('Show active-learning queries', value=True)
    al_query_n = st.number_input('Query top-n uncertain', min_value=1, max_value=200, value=20)
    al_run_query = st.button('Query uncertain samples (backend)')
    al_use_partial = st.checkbox('Prefer incremental retrain (SGD partial_fit) for large data', value=True)

    st.markdown('---')
    st.header('Geospatial Q&A (RAG)')
    rag_query = st.text_input('Ask a geospatial question (e.g. "Show me regions in Colombia with > 0.7 risk and near rivers.")')
    if st.button('Run RAG query'):
        try:
            # lightweight local retrieval-based answer
            import backend.rag as rag
            res = rag.answer_query(rag_query)
            st.write(res['text'])
            if res.get('path') and os.path.exists(res['path']):
                # show geojson on map (small overlay)
                try:
                    qg = gpd.read_file(res['path'])
                    m2 = draw_map(qg, qg.get('risk', [0]*len(qg)), [0]*len(qg))
                    st_folium(m2, width=600, height=400)
                except Exception as e:
                    st.write('Result saved to', res.get('path'))
        except Exception as e:
            st.error('RAG query failed: ' + str(e))


model = train_model(gdf, features)
# optionally load calibrated model from disk and use its predict_proba.
calibrated_model_path = None
try:
    import joblib
    metrics_path = os.path.join('reports', 'metrics.json')
    if use_calibrated and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        cal = metrics.get('calibration', {})
        cp = cal.get('calibrated_path')
        if cp and os.path.exists(cp):
            calibrated_model_path = cp
    # fallback: check models folder for any calibrated model
    if use_calibrated and calibrated_model_path is None:
        for fname in os.listdir('models'):
            if 'calibrated' in fname and fname.endswith('.pkl'):
                calibrated_model_path = os.path.join('models', fname)
                break
except Exception:
    calibrated_model_path = None

if calibrated_model_path:
    try:
        calib_model = joblib.load(calibrated_model_path)
        # use calibrated model's predict_proba by aligning DataFrame columns to the calibrated model
        df_calib_X = pd.get_dummies(gdf[features + ['land_use']]).reindex(columns=calib_model.feature_names_in_, fill_value=0)
        risk = calib_model.predict_proba(df_calib_X)[:, 1]
        # uncertainty still from uncalibrated RF if available (use same model)
        uncertainty = np.zeros(len(risk))
    except Exception:
        model = train_model(gdf, features)
        risk, uncertainty = predict_with_uncertainty(model, gdf, features)
else:
    risk, uncertainty = predict_with_uncertainty(model, gdf, features)

# load shap artifacts if requested
shap_feats, shap_vals, shap_top_csv = load_shap_artifacts()
def load_shap_summary(path='models/shap_summary.json'):
    if os.path.exists(path):
        import json
        with open(path, 'r') as f:
            return json.load(f)
    return None

shap_summary = load_shap_summary()


def ensure_job_queue():
    """Return True if job_queue is available; otherwise show a Streamlit warning and return False."""
    if job_queue is None:
        try:
            st.error('Background job support is unavailable: `backend.job_queue` could not be imported. Heavy tasks will run synchronously if attempted.')
        except Exception:
            print('Background job support is unavailable: backend.job_queue missing')
        return False
    return True

# Build navigation graph
G, tree = build_graph(gdf, risk, uncertainty, k=k_nn, risk_weight=risk_weight, uncert_weight=uncert_weight)


def compute_path_metrics(G, path_idx, risk, gdf):
    """Compute total cost, avg predicted risk along path, and length (km)."""
    if path_idx is None or len(path_idx) < 2:
        return None
    total_cost = 0.0
    total_dist = 0.0
    risks = []
    for u, v in zip(path_idx[:-1], path_idx[1:]):
        edge = G.get_edge_data(u, v)
        if edge is None:
            continue
        total_cost += edge.get('weight', 0.0)
        total_dist += edge.get('distance', 0.0)
        risks.append((risk[u] + risk[v]) / 2.0)
    avg_risk = float(np.mean(risks)) if len(risks) > 0 else 0.0
    length_km = float(total_dist / 1000.0)
    return {'total_cost': float(total_cost), 'avg_risk': float(avg_risk), 'length_km': float(length_km)}


# Note: main card already contains a Map header and container (map_container). Do not create a second placeholder.

# initialize persistent UI state
if 'start_idx' not in st.session_state:
    st.session_state.start_idx = None
if 'end_idx' not in st.session_state:
    st.session_state.end_idx = None
if 'path_A' not in st.session_state:
    st.session_state.path_A = None
if 'path_A_meta' not in st.session_state:
    st.session_state.path_A_meta = None
if 'path_B' not in st.session_state:
    st.session_state.path_B = None
if 'path_B_meta' not in st.session_state:
    st.session_state.path_B_meta = None

# Initial map render
map_obj = draw_map(gdf, risk, uncertainty, path_idx=st.session_state.get('path_A'), start_idx=st.session_state.start_idx, end_idx=st.session_state.end_idx, risk_threshold=risk_threshold)
map_obj = draw_map(gdf, risk, uncertainty, path_idx=st.session_state.get('path_A'), start_idx=st.session_state.start_idx, end_idx=st.session_state.end_idx, risk_threshold=risk_threshold)
# If there's a B path, draw both A and B
map_obj = draw_map(gdf, risk, uncertainty, path_idx=st.session_state.get('path_A'), start_idx=st.session_state.start_idx, end_idx=st.session_state.end_idx, risk_threshold=risk_threshold)

# helper to draw both paths (we'll manually add B overlay after creating base map)
def draw_all(map_obj, gdf, risk, uncertainty, pathA, pathB):
    if pathA is not None and len(pathA) > 1:
        coordsA = [[gdf.iloc[i].lat, gdf.iloc[i].lon] for i in pathA]
        folium.PolyLine(coordsA, color='blue', weight=4, opacity=0.9).add_to(map_obj)
    if pathB is not None and len(pathB) > 1:
        coordsB = [[gdf.iloc[i].lat, gdf.iloc[i].lon] for i in pathB]
        folium.PolyLine(coordsB, color='magenta', weight=4, opacity=0.9).add_to(map_obj)
    return map_obj

# show map
# render the map inside the main card container using the main_col context so it stays inside the card
map_obj = draw_all(map_obj, gdf, risk, uncertainty, st.session_state.get('path_A'), st.session_state.get('path_B'))
with main_col:
    map_data = st_folium(map_obj, width=700, height=700)
# close the main-card container so the map sits inside it visually
try:
    st.markdown('</div>', unsafe_allow_html=True)
except Exception:
    pass

# Handle map clicks
clicked = map_data.get('last_clicked') if isinstance(map_data, dict) else None
if clicked:
    click_lat = clicked.get('lat')
    click_lon = clicked.get('lng')
    if click_lat is not None and click_lon is not None:
        nearest = nearest_node(tree, gdf, click_lon, click_lat)
        st.info(f'Clicked nearest node: {nearest} (Risk={risk[nearest]:.3f}, Unc={uncertainty[nearest]:.3f})')
        # labeling
        if label_mode == 'Mark cleared (0)':
            # append to labels_delta instead of overwriting original data
            delta_path = 'data/labels_delta.csv'
            import csv
            os.makedirs(os.path.dirname(delta_path), exist_ok=True)
            # append index,label
            with open(delta_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nearest, 0])
            st.success(f'Node {nearest} labeled as CLEARED (0) (saved to labels_delta)')
        elif label_mode == 'Mark dangerous (1)':
            delta_path = 'data/labels_delta.csv'
            import csv
            os.makedirs(os.path.dirname(delta_path), exist_ok=True)
            with open(delta_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([nearest, 1])
            st.success(f'Node {nearest} labeled as DANGEROUS (1) (saved to labels_delta)')
        else:
            # selection for start/end (persist)
            if st.session_state.start_idx is None:
                st.session_state.start_idx = nearest
                st.success(f'Start selected: {st.session_state.start_idx}')
            elif st.session_state.end_idx is None:
                st.session_state.end_idx = nearest
                st.success(f'End selected: {st.session_state.end_idx}')

        # If we retrained or updated labels, recompute model & graph
        if retrain:
            model = train_model(gdf, features)
            risk, uncertainty = predict_with_uncertainty(model, gdf, features)
            G, tree = build_graph(gdf, risk, uncertainty, k=k_nn, risk_weight=risk_weight, uncert_weight=uncert_weight)

        # If SHAP available and user requested, show top contributors for clicked node
        if show_shap and shap_vals is not None and shap_feats is not None:
            try:
                row_idx = nearest
                contribs = shap_vals[row_idx]
                # pair feature names
                pairs = list(zip(shap_feats, contribs.tolist()))
                # sort by absolute contribution
                pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:10]
                st.write('Top SHAP contributions (clicked node)')
                for fname, val in pairs_sorted:
                    st.write(f"{fname}: {val:.4f}")
            except Exception as e:
                st.write('Failed to display SHAP for node:', e)

        # If start and end selected, compute shortest path
        if st.session_state.start_idx is not None and st.session_state.end_idx is not None:
            try:
                path_idx = nx.shortest_path(G, source=st.session_state.start_idx, target=st.session_state.end_idx, weight='weight')
                st.session_state.current_path = path_idx
                st.write(f'Found path of length {len(path_idx)} nodes')
            except Exception as e:
                st.error(f'Path search failed: {e}')

    # redraw map with updates and any saved A/B paths (render into main column)
    map_obj = draw_map(gdf, risk, uncertainty, path_idx=st.session_state.get('current_path'), start_idx=st.session_state.start_idx, end_idx=st.session_state.end_idx, risk_threshold=risk_threshold)
    map_obj = draw_all(map_obj, gdf, risk, uncertainty, st.session_state.get('path_A'), st.session_state.get('path_B'))
    with main_col:
        map_data = st_folium(map_obj, width=700, height=700)

# Buttons to save/compare paths (sidebar)
st.sidebar.markdown('---')
if st.sidebar.button('Save Current Path as "A"'):
    if 'current_path' in st.session_state and st.session_state.current_path is not None:
        st.session_state.path_A = st.session_state.current_path.copy()
        st.session_state.path_A_meta = {'risk_weight': risk_weight, 'uncert_weight': uncert_weight}
        st.success('Saved current path as A')
    else:
        st.warning('No current path to save. Select start/end first.')

st.sidebar.write('Comparison path B weights')
risk_weight_B = st.sidebar.slider('Risk weight (B)', 0.0, 20.0, float(risk_weight * 1.5))
uncert_weight_B = st.sidebar.slider('Uncertainty weight (B)', 0.0, 20.0, float(uncert_weight * 0.5))
if st.sidebar.button('Calculate & Compare as "B"'):
    if st.session_state.start_idx is not None and st.session_state.end_idx is not None:
        # build alternative graph with B weights
        G_B, tree_B = build_graph(gdf, risk, uncertainty, k=k_nn, risk_weight=risk_weight_B, uncert_weight=uncert_weight_B)
        try:
            path_B = nx.shortest_path(G_B, source=st.session_state.start_idx, target=st.session_state.end_idx, weight='weight')
            st.session_state.path_B = path_B
            st.session_state.path_B_meta = {'risk_weight': risk_weight_B, 'uncert_weight': uncert_weight_B}
            st.success('Computed path B')
            # compute metrics for A and B
            metaA = compute_path_metrics(G, st.session_state.get('path_A'), risk, gdf) if st.session_state.get('path_A') is not None else None
            metaB = compute_path_metrics(G_B, st.session_state.get('path_B'), risk, gdf)
            st.session_state.path_A_meta = metaA if metaA is not None else st.session_state.path_A_meta
            st.session_state.path_B_meta = metaB
        except Exception as e:
            st.error(f'Failed to compute path B: {e}')
    else:
        st.warning('Select start and end points first')

# Show metrics panel
with st.expander('Path Metrics', expanded=True):
    col1, col2 = st.columns(2)
    if st.session_state.get('path_A') is not None:
        meta = compute_path_metrics(G, st.session_state.get('path_A'), risk, gdf)
        col1.markdown('### Path A (Saved)')
        if meta:
            col1.write(f"Total Cost: {meta['total_cost']:.1f}")
            col1.write(f"Predicted Risk (avg): {meta['avg_risk']:.3f}")
            col1.write(f"Length: {meta['length_km']:.1f} km")
        else:
            col1.write('No metrics available')
    else:
        col1.write('Path A not set')

    if st.session_state.get('path_B') is not None:
        meta = compute_path_metrics(G if True else G_B, st.session_state.get('path_B'), risk, gdf)
        col2.markdown('### Path B (Comparison)')
        if meta:
            col2.write(f"Total Cost: {meta['total_cost']:.1f}")
            col2.write(f"Predicted Risk (avg): {meta['avg_risk']:.3f}")
            col2.write(f"Length: {meta['length_km']:.1f} km")
        else:
            col2.write('No metrics available')
    else:
        col2.write('Path B not set')

# Save dataset with risk & uncertainty for downstream use
save_with_risk(gdf, risk, uncertainty)

st.sidebar.markdown('---')
st.sidebar.write('Dataset saved to `data/terrain_data_with_risk.csv`')

# Active Learning panel (sidebar)
if al_show_queries:
    qf = 'models/active_queries.json'
    if al_run_query:
        # enqueue backend script to write queries (non-blocking)
        try:
            import sys
            args = [sys.executable, 'backend/active_learning.py', 'query', '--n', str(int(al_query_n))]
            jid = job_queue.enqueue_job(args, metadata={'type': 'active_learning_query', 'n': int(al_query_n)})
            st.success(f'Enqueued active-learning query job: {jid}')
        except Exception as e:
            st.write('Failed to enqueue backend query:', e)

    if os.path.exists(qf):
        try:
            aq = pd.read_json(qf)
            st.markdown('### Active Learning queries')
            st.dataframe(aq['queries'].apply(pd.Series))
            # allow quick labeling from queries
            sel = st.number_input('Label index (from top queries) to mark', min_value=0, max_value=100000, value=0)
            if st.button('Label selected query as DANGEROUS (1)'):
                try:
                    queries = aq['queries']
                    idx = int(queries[sel]['index'])
                    # persist label delta
                    import csv
                    delta_path = 'data/labels_delta.csv'
                    os.makedirs(os.path.dirname(delta_path), exist_ok=True)
                    with open(delta_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([idx, 1])
                    st.success(f'Labeled index {idx} as 1 (saved to labels_delta)')
                except Exception as e:
                    st.error('Failed to label:', e)
            if st.button('Retrain model from labels (active learning)'):
                # call retrain backend
                try:
                    import backend.active_learning as al
                    # pass preference for partial_fit by env var or parameter - we call retrain directly
                    al.retrain_with_labels()
                    st.success('Retrain triggered; model saved')
                    # reload model and recompute risk/uncertainty
                    model = train_model(gdf, features)
                    risk, uncertainty = predict_with_uncertainty(model, gdf, features)
                    G, tree = build_graph(gdf, risk, uncertainty, k=k_nn, risk_weight=risk_weight, uncert_weight=uncert_weight)
                except Exception as e:
                    st.error('Retrain failed:', e)
        except Exception as e:
            st.write('Failed to load active_queries.json:', e)
    else:
        st.write('No active query file found. Click "Query uncertain samples (backend)" to create one.')

# Background jobs list & logs (sidebar)
with st.sidebar.expander('Background jobs', expanded=False):
    try:
        jobs = job_queue.list_jobs()
        if not jobs:
            st.write('No jobs found')
        else:
            for j in jobs:
                jid = j.get('id')
                status = j.get('status')
                with st.expander(f"{jid} — {status}", expanded=False):
                    st.write('Command:', j.get('command'))
                    st.write('Metadata:', j.get('metadata'))
                    if st.button('Refresh log', key=f'refresh_log_{jid}'):
                        pass
                    if st.button('View log', key=f'view_log_{jid}'):
                        log = job_queue.read_log(jid)
                        st.text_area(f'Log for {jid}', value=log, height=240)
    except Exception as e:
        st.write('Failed to load job list:', e)

# Small utility: enqueue a lightweight test job from the UI
if 'enqueue_test' not in st.session_state:
    st.session_state.enqueue_test = False
if st.sidebar.button('Enqueue test job'):
    if ensure_job_queue():
        try:
            import sys
            args = [sys.executable, '-c', 'print("Hello from test job")']
            jid = job_queue.enqueue_job(args, metadata={'type': 'test_job'})
            st.success(f'Enqueued test job: {jid}')
        except Exception as e:
            st.error(f'Failed to enqueue test job: {e}')

# SHAP compute trigger (enqueue job — heavy task)
if run_shap_compute:
    try:
        import sys
        args = [sys.executable, 'backend/shap_explain.py']
        if shap_sample_size and int(shap_sample_size) > 0:
            args += ['--sample-size', str(int(shap_sample_size))]
        jid = job_queue.enqueue_job(args, metadata={'type': 'shap_compute', 'sample_size': int(shap_sample_size) if shap_sample_size else None})
        st.success(f'Enqueued SHAP compute job: {jid} — start the worker (python backend/worker.py) to process it')
    except Exception as e:
        st.error('Failed to enqueue SHAP compute job:', e)

# Fast SHAP summary display
if shap_summary is not None and show_shap:
    try:
        st.markdown('### SHAP summary — mean |SHAP| per feature (cached)')
        # show top 20 features
        items = list(shap_summary.items())[:20]
        import pandas as _pd
        df_sh = _pd.DataFrame(items, columns=['feature', 'mean_abs_shap'])
        st.bar_chart(df_sh.set_index('feature'))
    except Exception as e:
        st.write('Failed to render SHAP summary:', e)


### Model evaluation panel (loads backend reports if available)
import json
REPORTS_DIR = 'reports'
metrics_path = f'{REPORTS_DIR}/metrics.json'
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        with st.expander('Model Evaluation (reports)', expanded=False):
            st.write('Best model:', metrics.get('best_model'))
            st.write('Best test ROC-AUC:', metrics.get('best_score_test_roc_auc'))
            cols = st.columns(3)
            plots = metrics.get('plots', {})
            if 'roc' in plots and os.path.exists(plots['roc']):
                cols[0].image(plots['roc'], caption='ROC Curve')
            if 'pr' in plots and os.path.exists(plots['pr']):
                cols[1].image(plots['pr'], caption='PR Curve')
            if 'calibration' in plots and os.path.exists(plots['calibration']):
                cols[2].image(plots['calibration'], caption='Calibration')
            st.write('Model CV and test metrics:')
            st.json(metrics.get('models', {}))
            # load operating points if present
            op_csv = os.path.join(REPORTS_DIR, 'operating_points.csv')
            if os.path.exists(op_csv):
                df_op = pd.read_csv(op_csv)
                st.markdown('#### Operating points (thresholds)')
                st.dataframe(df_op.head(20))
                st.download_button('Download operating_points.csv', data=open(op_csv, 'rb'), file_name='operating_points.csv')
    except Exception:
        st.write('Failed to load reports/metrics.json')
else:
    with st.expander('Model Evaluation (reports)', expanded=False):
        st.write('No reports found. Run `python backend/train_baselines.py` to generate evaluation reports.')
