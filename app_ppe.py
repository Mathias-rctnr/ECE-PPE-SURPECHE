import streamlit as st
import dask.dataframe as dd
import pandas as pd
import folium
import os
import numpy as np
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

# 📂 Dossier contenant les fichiers CSV
DATA_DIR = "c:/Users/mcore/OneDrive/Desktop/ING4-Finance/PPE/DATA_DIR"

# 📍 Coordonnées de Fécamp
FECAMP_LAT = 49.7570
FECAMP_LON = 0.3746

# 📌 Définition des limites de la carte (zone Manche)
BOUNDS = [[49.0, -2.0], [51.0, 2.0]]  # [Sud-Ouest, Nord-Est]

# -----------------------------------------------------------------------------
# Fonction de calcul de distance (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# -----------------------------------------------------------------------------
# Fonction pour créer un quadrillage sur la zone définie par BOUNDS
def create_grid(bounds, n_rows, n_cols):
    lat_min, lon_min = bounds[0]
    lat_max, lon_max = bounds[1]
    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols
    grid_cells = []
    for i in range(n_rows):
        for j in range(n_cols):
            cell_bounds = [
                [lat_min + i * lat_step, lon_min + j * lon_step],         # coin sud-ouest
                [lat_min + (i+1) * lat_step, lon_min + (j+1) * lon_step]      # coin nord-est
            ]
            grid_cells.append({
                "id": f"{i},{j}",
                "bounds": cell_bounds
            })
    return grid_cells

# -----------------------------------------------------------------------------
# Chargement des fichiers CSV
csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
if not csv_files:
    st.error("Aucun fichier CSV trouvé dans le dossier " + DATA_DIR)
    st.stop()

st.sidebar.header("📅 Sélection de la date")
selected_file = st.sidebar.selectbox("📌 Choisir un fichier", csv_files)
file_path = os.path.join(DATA_DIR, selected_file)

@st.cache_data
def load_data(file_path):
    df = dd.read_csv(file_path, dtype={'fishing_hours': 'float64'})
    # Calculer la distance par rapport à Fécamp
    df["distance_km"] = haversine(df["cell_ll_lat"], df["cell_ll_lon"], FECAMP_LAT, FECAMP_LON)
    # Par défaut, on peut filtrer par un rayon assez grand (la totalité de BOUNDS)
    df = df[df["distance_km"] <= 150]
    return df.compute()

df = load_data(file_path)
if df.empty:
    st.warning("⚠️ Aucun bateau détecté dans la zone.")
    st.stop()

# -----------------------------------------------------------------------------
# Filtres Streamlit pour les données des bateaux
st.sidebar.header("🎛️ Filtres avancés")
selected_geartype = st.sidebar.multiselect(
    "🛥️ Sélectionner le type de bateau",
    options=df["geartype"].unique(),
    default=list(df["geartype"].unique()),
)
selected_flag = st.sidebar.multiselect(
    "🚩 Sélectionner le pays d'origine",
    options=df["flag"].unique(),
    default=list(df["flag"].unique()),
)
fishing_time_threshold = st.sidebar.slider(
    "⏳ Filtrer par temps de pêche (heures)",
    0.0,
    float(df["fishing_hours"].max()),
    (0.0, float(df["fishing_hours"].max()))
)
filtered_df = df[
    (df["geartype"].isin(selected_geartype)) &
    (df["flag"].isin(selected_flag)) &
    (df["fishing_hours"].between(fishing_time_threshold[0], fishing_time_threshold[1]))
]

# -----------------------------------------------------------------------------
# Paramètres du quadrillage
st.sidebar.header("📍 Configuration du quadrillage")
n_rows = st.sidebar.number_input("Nombre de lignes du quadrillage", min_value=1, max_value=20, value=4)
n_cols = st.sidebar.number_input("Nombre de colonnes du quadrillage", min_value=1, max_value=20, value=4)
grid_cells = create_grid(BOUNDS, n_rows, n_cols)

# Sélection des cellules sensibles (identifiées par leur indice "ligne,colonne")
sensitive_cells = st.sidebar.multiselect(
    "Sélectionner les cellules sensibles (ex: 0,1)",
    [cell["id"] for cell in grid_cells]
)

# -----------------------------------------------------------------------------
# Création de la carte avec Folium
st.subheader("🗺️ Carte avec Quadrillage")
# Pour recentrer la carte, ajustez la position de départ si besoin
m = folium.Map(
    location=[FECAMP_LAT, FECAMP_LON + 0.1],
    zoom_start=8,
    max_bounds=True,
    min_lat=BOUNDS[0][0], max_lat=BOUNDS[1][0],
    min_lon=BOUNDS[0][1], max_lon=BOUNDS[1][1]
)

# Ajout des marqueurs pour les bateaux dans un MarkerCluster
marker_cluster = MarkerCluster().add_to(m)
for _, row in filtered_df.iterrows():
    icon = folium.Icon(color="blue", icon="ship", prefix="fa")
    folium.Marker(
        location=[row["cell_ll_lat"], row["cell_ll_lon"]],
        popup=folium.Popup(f"""
            <b>🛥️ Type :</b> {row['geartype']}<br>
            <b>🚩 Pays :</b> {row['flag']}<br>
            <b>⏳ Temps de pêche :</b> {row['fishing_hours']}h<br>
            <b>📏 Distance :</b> {row['distance_km']:.2f} km
        """, max_width=250),
        tooltip=f"{row['geartype']} - {row['flag']}",
        icon=icon
    ).add_to(marker_cluster)

# Tracer le quadrillage et mettre en évidence les cellules sensibles
for cell in grid_cells:
    # Si la cellule est définie comme sensible, on la colorie en rouge
    color = "red" if cell["id"] in sensitive_cells else "blue"
    folium.Rectangle(
        bounds=cell["bounds"],
        color=color,
        weight=1,
        fill=True,
        fill_opacity=0.2
    ).add_to(m)

# Optionnel : Afficher une heatmap si besoin
if st.sidebar.checkbox("Afficher la Heatmap", value=False):
    heat_data = [[row["cell_ll_lat"], row["cell_ll_lon"]] for index, row in filtered_df.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)

# Afficher la carte
st_folium(m, width=1200, height=800)

