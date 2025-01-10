import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Dictionnaire des couleurs
pal = {
    "NC": (0, 0, 0), "BJ": (255, 255, 255),
    "JO": (228, 189, 104), "BC": (0, 134, 214),
    "VL": (174, 150, 212), "VG": (63, 142, 67),
    "RE": (222, 67, 67), "BM": (0, 120, 191),
    "OM": (249, 153, 99), "VGa": (59, 102, 94),
    "BG": (163, 216, 225), "VM": (236, 0, 140),
    "GA": (166, 169, 170), "VB": (94, 67, 183), "BF": (4, 47, 86),
}

# Listes de palettes fixes
palettes = [
    ["NC", "RE", "JO", "BJ"],
    ["NC", "BM", "BG", "BJ"],
    ["NC", "BM", "JO", "BJ"],
    ["NC", "VB", "OM", "BJ"],
]

# Ajouter des palettes à 6 couleurs
palettes_6 = [
    ["NC", "VB", "RE", "OM", "JO", "BJ"],
    ["NC", "BF", "BM", "BC", "BG", "BJ"],
    ["NC", "VGa", "BM", "GA", "JO", "BJ"],
    ["NC", "BF", "VGa", "VG", "VL", "BJ"],
]

st.title("Tylice Simplifié")

# Initialiser les états des boutons dans st.session_state
if "mode" not in st.session_state:
    st.session_state.mode = "4"  # Par défaut 4 couleurs

def set_mode(mode):
    """Définir le mode actif et désactiver les autres."""
    st.session_state.mode = mode

# Boutons pour sélectionner le mode avec état "enfoncé"
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("4 Couleurs : 7.95 €", key="4_button",
                 disabled=st.session_state.mode == "4"):
        set_mode("4")
with col2:
    if st.button("6 Couleurs : 12.95 €", key="6_button",
                 disabled=st.session_state.mode == "6"):
        set_mode("6")
with col3:
    if st.button("Personnalisation avancée", key="advanced_button",
                 disabled=st.session_state.mode == "advanced"):
        set_mode("advanced")
with col4:
    if st.button("Exemples", key="examples_button",
                 disabled=st.session_state.mode == "examples"):
        set_mode("examples")

# Comportement selon le mode sélectionné
if st.session_state.mode == "4":
    st.write("Mode 4 Couleurs sélectionné.")
    # Ajoutez ici le traitement pour le mode 4 couleurs.
elif st.session_state.mode == "6":
    st.write("Mode 6 Couleurs sélectionné.")
    # Ajoutez ici le traitement pour le mode 6 couleurs.
elif st.session_state.mode == "advanced":
    st.write("Mode Personnalisation avancée sélectionné.")
    # Ajoutez ici le traitement pour la personnalisation avancée.
elif st.session_state.mode == "examples":
    st.write("Mode Exemples sélectionné.")
    # Ajoutez ici l'affichage des exemples.
