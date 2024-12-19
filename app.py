import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Palette de couleurs
pal = {
    "Noir": (0, 0, 0), "Blanc": (255, 255, 255),
    "Or": (228, 189, 104), "Cyan": (0, 134, 214),
    "Lila": (174, 150, 212), "Vert": (63, 142, 67),
    "Rouge": (222, 67, 67), "Bleu": (0, 120, 191),
    "Orange": (249, 153, 99), "Vert foncé": (59, 102, 94),
    "Bleu clair": (163, 216, 225), "Magenta": (236, 0, 140),
    "Argent": (166, 169, 170), "Violet": (94, 67, 183),
    "Bleu foncé": (4, 47, 86),
}

# Fonction pour convertir une couleur en HTML
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

# Initialiser la sélection des couleurs
if "selected_colors" not in st.session_state:
    st.session_state.selected_colors = []

# Sélectionner une couleur
def add_color_to_selection(color_name):
    if color_name in pal:  # Vérifie que la couleur est bien dans la palette
        if len(st.session_state.selected_colors) < st.session_state.num_selections:
            if color_name not in st.session_state.selected_colors:
                st.session_state.selected_colors.append(color_name)

# Configuration de l'interface
st.title("Tylice - Sélection des Couleurs")
uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if "num_selections" not in st.session_state:
    st.session_state.num_selections = 4

if st.button("4 Couleurs : 7.95 €"):
    st.session_state.num_selections = 4
if st.button("6 Couleurs : 11.95 €"):
    st.session_state.num_selections = 6

# Afficher la palette de couleurs
st.markdown("### Cliquez sur les couleurs pour les sélectionner :")
