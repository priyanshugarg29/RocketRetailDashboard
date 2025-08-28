# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 1. Configuration  
DATA_DIR      =  Path(__file__).parent.parent
FIGS_DIR      = f"{DATA_DIR}/figs"  
TABLES_DIR    = f"{DATA_DIR}/data"  
SHAP_DIR      = f"{DATA_DIR}/shap"  
SESSION_INDEX = pd.read_csv(f"{TABLES_DIR}/20250827_195718_dataset_overview.csv")["sessions"]  # adjust  

st.set_page_config(page_title="Rocket Retail Segmentation", layout="wide")

# 2. Sidebar controls  
st.sidebar.title("Controls")  
model_list = pd.read_csv(f"{TABLES_DIR}/20250827_195718_eval_internal_metrics.csv")["model"].tolist()  
lead_model = st.sidebar.selectbox("Select clustering model:", model_list)  
show_shap  = st.sidebar.checkbox("Show SHAP explanations", value=False)  
time_split = st.sidebar.checkbox("Show temporal drift", value=False)

# 3. EDA Summary  
st.header("Dataset Overview")  
overview = pd.read_csv(f"{TABLES_DIR}/20250827_195718_dataset_overview.csv")  
st.metric("Total Events", f"{int(overview.rows):,}")  
st.metric("Total Sessions", f"{int(overview.sessions):,}")  
st.metric("Unique Visitors", f"{int(overview.visitors):,}")  

# Event distribution  
evt = pd.read_csv(f"{TABLES_DIR}/event_distribution.csv")  
fig_evt = px.bar(evt, x="event", y="count", title="Event Distribution")  
st.plotly_chart(fig_evt, use_container_width=True)

# 4. Temporal Patterns  
st.header("Temporal Patterns")  
hr = pd.read_csv(f"{TABLES_DIR}/temporal_hour_event_counts.csv")  
fig_hr = px.line(hr, x="hour", y="count", color="event", title="Hourly Event Volume")  
st.plotly_chart(fig_hr, use_container_width=True)  
dow = pd.read_csv(f"{TABLES_DIR}/temporal_dow_event_counts.csv")  
fig_dow = px.line(dow, x="dow_label", y="count", color="event", title="Weekly Event Volume")  
st.plotly_chart(fig_dow, use_container_width=True)

# 5. Session Funnel  
st.header("Session Funnel")  
funnel = pd.read_csv(f"{TABLES_DIR}/funnel_session_level.csv")  
fig_funnel = px.bar(funnel, x="stage", y="sessions_reached", text="sessions_reached",  
                    title="Session-level Funnel")  
fig_funnel.update_traces(texttemplate="%{text:,}", textposition="outside")  
st.plotly_chart(fig_funnel, use_container_width=True)

# 6. UMAP Projection  
st.header("UMAP Visualization")  
coords = pd.read_csv(f"{TABLES_DIR}/20250827_195718_umap_all_models_coords.csv")  

if lead_model == "rfm_proxy_kmeans_session_12":
    labels = pd.read_csv(f"{TABLES_DIR}/20250827_195718_session_rfm_proxy_labels_k12.csv")
    if "label" in labels.columns:
        labels = labels.rename(columns={"label": lead_model})
else:
    labels = pd.read_csv(f"{TABLES_DIR}/20250827_195718_cluster_labels_all_models.csv")


df_umap = coords.merge(labels[["session_id", lead_model]], on="session_id")  
fig_umap = px.scatter(df_umap, x="umap_x", y="umap_y", color=lead_model,  
                      title=f"UMAP – {lead_model}", width=800, height=600, opacity=0.6)  
st.plotly_chart(fig_umap, use_container_width=True)


# 7. Internal Metrics Comparison  
st.header("Model Evaluation Metrics")  
metrics = pd.read_csv(f"{TABLES_DIR}/20250827_195718_eval_internal_metrics.csv")  
st.dataframe(metrics.style.format({"silhouette":"{:.4f}",  
                                   "calinski_harabasz":"{:.0f}",  
                                   "davies_bouldin":"{:.4f}"}))

# 8. Cross-Model Agreement (ARI / AMI)  
st.header("Cross-Model Agreement")  
ari = pd.read_csv(f"{TABLES_DIR}/20250827_195718_eval_cross_model_ARI.csv", index_col=0)  
ami = pd.read_csv(f"{TABLES_DIR}/20250827_195718_eval_cross_model_AMI.csv", index_col=0)  
st.subheader("Adjusted Rand Index (ARI)")  
st.dataframe(ari)  
st.subheader("Adjusted Mutual Information (AMI)")  
st.dataframe(ami)

# 9. Temporal Drift (optional)  
if time_split:  
    st.header("Cluster Temporal Drift")  
    drift = pd.read_csv(f"{TABLES_DIR}/20250827_195718_eval_temporal_drift.csv")  
    st.dataframe(drift.sort_values("abs_diff", ascending=False).head(10))

# 10. Segment Personas  
st.header("Segment Personas")  
persona = pd.read_csv(f"{DATA_DIR}/20250827_195718_session_rfm_proxy_profile_k12.csv")  
st.dataframe(persona.style.format({"share":"{:.2%}",  
                                   "tx_rate":"{:.2%}",  
                                   "atc_rate":"{:.2%}"}))

# 11. SHAP Explanations (optional)  
if show_shap:  
    st.header("SHAP Feature Importance")  
    shap_feats = pd.read_csv(f"{SHAP_DIR}/shap_sample_class_sizes_{lead_model}.csv")  # example  
    st.dataframe(shap_feats)  
    shap_img = f"{FIGS_DIR}/{lead_model}_shap_beeswarm.png"  # adjust naming  
    st.image(shap_img, caption=f"SHAP Beeswarm – {lead_model}", use_column_width=True)

# 12. Footer  
st.markdown("---")  
st.markdown("* 2025 Rocket Retail Segmentation Dashboard")

