import streamlit as st
import dask.dataframe as dd
import pandas as pd
import folium
import os
import numpy as np
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from datetime import timedelta

# ---------------------------
# Paramètres généraux
# ---------------------------
# Dossier contenant les fichiers CSV
DATA_DIR = "c:/Users/mcore/OneDrive/Desktop/ING4-Finance/PPE/DATA_DIR"

# Définir les limites de la Manche (Sud-Ouest et Nord-Est)
BOUNDS = [[49.0, -2.0], [51.0, 2.0]]  # [lat_min, lon_min] et [lat_max, lon_max]

# Pour le tracé de la carte, on centre sur la Manche
CENTER = [50.0, 0.0]

# Grille fixe : nombre de lignes et de colonnes (carreaux)
N_ROWS = 20
N_COLS = 20


# ---------------------------
# Fonctions utilitaires
# ---------------------------
def haversine(lat1, lon1, lat2, lon2):
    # Calcul de la distance en km entre deux points
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def create_grid(bounds, n_rows, n_cols):
    # Crée une grille fixe couvrant la zone définie par BOUNDS
    lat_min, lon_min = bounds[0]
    lat_max, lon_max = bounds[1]
    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols
    grid_cells = []
    for i in range(n_rows):
        for j in range(n_cols):
            cell_bounds = [
                [lat_min + i * lat_step, lon_min + j * lon_step],  # coin sud-ouest
                [lat_min + (i + 1) * lat_step, lon_min + (j + 1) * lon_step]  # coin nord-est
            ]
            grid_cells.append({
                "id": f"{i},{j}",
                "bounds": cell_bounds
            })
    return grid_cells


# ---------------------------
# Chargement des données
# ---------------------------
# Liste des fichiers CSV disponibles
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
    # Filtrage par coordonnées pour ne conserver que les données dans la Manche
    df = df[(df["cell_ll_lat"] >= BOUNDS[0][0]) & (df["cell_ll_lat"] <= BOUNDS[1][0]) &
            (df["cell_ll_lon"] >= BOUNDS[0][1]) & (df["cell_ll_lon"] <= BOUNDS[1][1])]
    # Calcul de la distance par rapport au centre (50,0) pour information (optionnel)
    df["distance_km"] = haversine(df["cell_ll_lat"], df["cell_ll_lon"], CENTER[0], CENTER[1])
    return df.compute()


df = load_data(file_path)
if df.empty:
    st.warning("⚠️ Aucun bateau détecté dans la zone de la Manche.")
    st.stop()

# ---------------------------
# Filtres et configuration
# ---------------------------
st.sidebar.header("🎛️ Filtres avancés")
selected_geartype = st.sidebar.multiselect(
    "🛥️ Sélectionner le type de bateau",
    options=df["geartype"].unique(),
    default=list(df["geartype"].unique())
)
selected_flag = st.sidebar.multiselect(
    "🚩 Sélectionner le pays d'origine",
    options=df["flag"].unique(),
    default=list(df["flag"].unique())
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

# Configuration de la grille fixe
grid_cells = create_grid(BOUNDS, N_ROWS, N_COLS)

# Sélection des cellules sensibles dans la grille
sensitive_cells = st.sidebar.multiselect(
    "Sélectionner les cellules sensibles (ex: 0,1)",
    [cell["id"] for cell in grid_cells]
)

# Configuration des temps de restriction par type de bateau
st.sidebar.header("⚙️ Restrictions par type de bateau")
allowed_times = {}
for boat_type in sorted(filtered_df["geartype"].unique()):
    allowed_times[boat_type] = st.sidebar.number_input(
        f"Temps max autorisé pour {boat_type} (heures)",
        min_value=0.0, value=1.0, step=0.1
    )

# ---------------------------
# Création de la carte
# ---------------------------
st.subheader("🗺️ Carte de la Manche avec Grille")
# Carte centrée sur la Manche, sans recadrage automatique
m = folium.Map(
    location=CENTER,
    zoom_start=8
)

# Ajout des marqueurs des bateaux
marker_cluster = MarkerCluster().add_to(m)
for _, row in filtered_df.iterrows():
    icon = folium.Icon(color="blue", icon="ship", prefix="fa")
    folium.Marker(
        location=[row["cell_ll_lat"], row["cell_ll_lon"]],
        popup=folium.Popup(f"""
            <b>🛥️ Type :</b> {row['geartype']}<br>
            <b>🚩 Pays :</b> {row['flag']}<br>
            <b>⏳ Temps de pêche :</b> {row['fishing_hours']}h<br>
            <b>📏 Dist. du centre :</b> {row['distance_km']:.2f} km
        """, max_width=250),
        tooltip=f"{row['geartype']} - {row['flag']}",
        icon=icon
    ).add_to(marker_cluster)

# Tracé de la grille fixe
for cell in grid_cells:
    color = "red" if cell["id"] in sensitive_cells else "blue"
    folium.Rectangle(
        bounds=cell["bounds"],
        color=color,
        weight=1,
        fill=True,
        fill_opacity=0.2
    ).add_to(m)

# Optionnel : Heatmap si souhaité
if st.sidebar.checkbox("Afficher la Heatmap", value=False):
    heat_data = [[row["cell_ll_lat"], row["cell_ll_lon"]] for index, row in filtered_df.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)

st_folium(m, width=1200, height=800)

# ---------------------------
# Calcul et affichage des alertes de dépassement
# ---------------------------
# Vérifier que les colonnes nécessaires existent
if "vessel_id" in filtered_df.columns and "timestamp" in filtered_df.columns:
    filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"])

    # Affecter à chaque observation une cellule de la grille fixe
    lat_min, lon_min = BOUNDS[0]
    lat_max, lon_max = BOUNDS[1]
    lat_step = (lat_max - lat_min) / N_ROWS
    lon_step = (lon_max - lon_min) / N_COLS


    def assign_cell(row):
        i = int((row["cell_ll_lat"] - lat_min) // lat_step)
        j = int((row["cell_ll_lon"] - lon_min) // lon_step)
        i = min(i, N_ROWS - 1)
        j = min(j, N_COLS - 1)
        return f"{i},{j}"


    filtered_df["cell_id"] = filtered_df.apply(assign_cell, axis=1)

    # Groupement par bateau et par cellule pour calculer le temps passé
    dwell = filtered_df.groupby(["vessel_id", "cell_id"]).agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        geartype=("geartype", "first"),
        last_lat=("cell_ll_lat", "last"),
        last_lon=("cell_ll_lon", "last")
    )
    dwell["dwell_duration"] = dwell["end_time"] - dwell["start_time"]

    # Ajout du temps autorisé selon la nature du bateau
    # (conversion en timedelta)
    dwell["allowed_timedelta"] = dwell["geartype"].apply(lambda gt: timedelta(hours=allowed_times.get(gt, 1.0)))
    dwell["overrun"] = dwell["dwell_duration"] - dwell["allowed_timedelta"]

    # On ne considère que les alertes dans les cellules sensibles
    alerts = dwell[(dwell["overrun"] > timedelta(0)) &
                   (dwell.index.get_level_values("cell_id").isin(sensitive_cells))].reset_index()

    st.subheader("🚨 Alertes de dépassement")
    if not alerts.empty:
        # Affichage de la liste des bateaux en dépassement avec position et temps dépassé
        alerts["overrun_str"] = alerts["overrun"].astype(str)
        alerts["dwell_str"] = alerts["dwell_duration"].astype(str)
        alerts["allowed_str"] = alerts["allowed_timedelta"].astype(str)
        st.dataframe(alerts[["vessel_id", "cell_id", "geartype", "last_lat", "last_lon", "dwell_str", "allowed_str",
                             "overrun_str"]])
    else:
        st.success("Aucun bateau n'a dépassé son temps autorisé dans une zone sensible.")
else:
    st.warning("Les colonnes 'vessel_id' et/ou 'timestamp' ne sont pas disponibles pour le calcul des dépassements.")
