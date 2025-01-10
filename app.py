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

# Initialiser l'état du bouton "Personnalisation avancée"
if "advanced_button_state" not in st.session_state:
    st.session_state.advanced_button_state = False

# Initialiser le mode par défaut
if "mode" not in st.session_state:
    st.session_state.mode = "4"

# Texte du bouton basé sur l'état
button_text = "Exemples" if st.session_state.advanced_button_state else "Personnalisation avancée"

# Boutons pour sélectionner le mode
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("4 Couleurs : 7.95 €"):
        st.session_state.mode = "4"
with col2:
    if st.button("6 Couleurs : 12.95 €"):
        st.session_state.mode = "6"
with col3:
    if st.button(button_text):
        st.session_state.advanced_button_state = not st.session_state.advanced_button_state
        # Mettre à jour le mode si nécessaire
        st.session_state.mode = "advanced" if st.session_state.advanced_button_state else "4"

# Affichage du mode sélectionné
if st.session_state.mode == "4":
    st.write("Mode 4 couleurs sélectionné.")
elif st.session_state.mode == "6":
    st.write("Mode 6 couleurs sélectionné.")
elif st.session_state.mode == "advanced":
    st.write("Mode personnalisation avancée activé.")
