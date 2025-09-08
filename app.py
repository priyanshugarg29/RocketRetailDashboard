# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 1. Configuration  
DATA_DIR      =  Path(__file__).parent.parent
FIGS_DIR      = f"{DATA_DIR}/figs"  
TABLES_DIR    = f"{DATA_DIR}/data"  
SHAP_DIR      = f"{DATA_DIR}/data"  
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
st.markdown("""
This demonstrates the size and richness of Rocket Retail’s dataset by accounting for total events (e.g., clicks, views), sessions, and unique customers. 
This confirms that the downstream insights are built on a robust and representative sample, reaffirming confidence in the following findings.
""")

# Event distribution  
evt = pd.read_csv(f"{TABLES_DIR}/event_distribution.csv")  
fig_evt = px.bar(evt, x="event", y="count", title="Event Distribution")  
st.plotly_chart(fig_evt, use_container_width=True)
st.markdown("""Above event distribution chart indicates how customer activity is distributed, mainly with high count of product views, while having fewer add-to-carts and purchases events.
Since, most visitors browse without advancing further, it highlights the significance of boosting conversion using targeted strategies to move more customers down the purchase funnel.
""")

# 4. Temporal Patterns  
st.header("Temporal Patterns")  
st.markdown("""Below temporal patterns reveal when the website is most and least active, helping managers align campaigns and support resources with peak demand times, maximizing operational efficiency and marketing impact.""")
hr = pd.read_csv(f"{TABLES_DIR}/temporal_hour_event_counts.csv")  
fig_hr = px.line(hr, x="hour", y="count", color="event", title="Hourly Event Volume")  
st.plotly_chart(fig_hr, use_container_width=True) 
st.markdown(""" Above graph shows the daily clickstream activity flows rhythm of Rocket Retail’s customers across each hour.
Most visitors seems to engage with the website during the late afternoon and evening, as evident by the steep rise in product views starting around 15:00, peaking around 18:00–21:00. 
This is when shoppers are actively browsing, likely after work or during their leisure time. However, while browsing is frequent, the number of add-to-cart actions and completed transactions remains much lower (close to the baseline) throughout the day. This highlights how only a small fraction of visitors move beyond looking at products to actually starting or completing a purchase.
This pattern demonstrates a valuable story for business leaders as marketing campaigns, product launches, and customer support are likely to have the greatest impact if timed for these afternoon and evening peaks. While, the quieter mid-morning and midday periods offer opportunities for maintenance work or targeted experiments without interfering with core customer engagement windows.""")

dow = pd.read_csv(f"{TABLES_DIR}/temporal_dow_event_counts.csv")  
fig_dow = px.line(dow, x="dow_label", y="count", color="event", title="Weekly Event Volume")  
st.plotly_chart(fig_dow, use_container_width=True)
st.markdown("""Above graph shows how customer activity changes over the week on Rocket Retail. Most people visit and view products at the start of the week, with Monday and Tuesday being the busiest days for browsing. As the week goes on, the number of views slowly drops, reaching the lowest point on Saturday, before going up again a little on Sunday.
The number of customers adding items to their cart or making purchases stays low throughout the week, much lower than the number of people just viewing items. This means that while many shoppers are interested early in the week, only a few actually buy.
For the business, this pattern suggests that running deals or campaigns at the start of the week could reach more active shoppers. Special weekend offers might also help turn more browsers into buyers when activity picks up again on Sunday. Understanding these weekly trends helps managers plan promotions to match when customers are most likely to engage.
""")

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

# 9. Temporal Drift
if time_split:  
    st.header("Cluster Temporal Drift")  
    drift = pd.read_csv(f"{TABLES_DIR}/20250827_195718_eval_temporal_drift.csv")  
    st.dataframe(drift.sort_values("abs_diff", ascending=False).head(10))

# 10. Segment Personas  
st.header("Segment Personas")  
persona = pd.read_csv(f"{SHAP_DIR}/20250827_195718_session_rfm_proxy_profile_k12.csv")  
st.dataframe(persona.style.format({"share":"{:.2%}",  
                                   "tx_rate":"{:.2%}",  
                                   "atc_rate":"{:.2%}"}))

# 11. SHAP Explanations
if show_shap:  
    st.header("SHAP Feature Importance")  
    shap_feats = pd.read_csv(f"{SHAP_DIR}/shap_sample_class_sizes_agg_ward_12.csv")   
    st.dataframe(shap_feats)  
    shap_img = f"{FIGS_DIR}/shap_beeswarm_agg_ward_12.png" 
    st.image(shap_img, caption=f"SHAP Beeswarm – ward linkage agglomerative clustering", use_column_width=True)

# 12. Footer  
st.markdown("---")  
st.markdown("* 2025 Rocket Retail Segmentation Dashboard")

