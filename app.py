import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import base64

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

# Configuration du style CSS pour une grille de couleurs
css = """
    <style>
        .color-box {
            display: inline-block;
            margin: 10px;
            width: 50px;
            height: 50px;
            border: 2px solid black;
            cursor: pointer;
            border-radius: 5px;
        }
        .color-box:hover {
            border: 4px solid black;
        }
        .selected-color {
            border: 4px solid gold !important;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Titre de l'application
st.title("Tylice - Sélection des Couleurs")

# Chargement de l'image
uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

# Sélection du nombre de couleurs
if "num_selections" not in st.session_state:
    st.session_state.num_selections = 4

if st.button("4 Couleurs : 7.95 €"):
    st.session_state.num_selections = 4
if st.button("6 Couleurs : 11.95 €"):
    st.session_state.num_selections = 6

num_selections = st.session_state.num_selections

# Initialiser les couleurs sélectionnées
if "selected_colors" not in st.session_state:
    st.session_state.selected_colors = []

# Gestion des clics sur les couleurs
def add_color_to_selection(color_name):
    if len(st.session_state.selected_colors) < num_selections:
        st.session_state.selected_colors.append(color_name)

# Afficher la palette de couleurs dans une grille
st.markdown("### Cliquez sur les couleurs à sélectionner :")
cols = st.columns(6)
for i, (color_name, color_rgb) in enumerate(pal.items()):
    with cols[i % 6]:
        color_hex = rgb_to_hex(color_rgb)
        btn_clicked = st.button(
            f" ",
            key=f"color_button_{color_name}",
            help=color_name,
        )
        if btn_clicked:
            add_color_to_selection(color_name)

        # Style dynamique pour montrer les couleurs sélectionnées
        selected_style = "selected-color" if color_name in st.session_state.selected_colors else ""
        st.markdown(
            f'<div class="color-box {selected_style}" style="background-color: {color_hex};"></div>',
            unsafe_allow_html=True,
        )

# Afficher les couleurs sélectionnées
if st.session_state.selected_colors:
    st.markdown("### Couleurs sélectionnées :")
    cols = st.columns(len(st.session_state.selected_colors))
    for i, color_name in enumerate(st.session_state.selected_colors):
        color_rgb = pal[color_name]
        color_hex = rgb_to_hex(color_rgb)
        with cols[i]:
            st.markdown(
                f'<div class="color-box" style="background-color: {color_hex};"></div>',
                unsafe_allow_html=True,
            )

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    img_arr = np.array(image)
    pixels = img_arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_selections, random_state=0).fit(pixels)
    labels = kmeans.labels_

    # Recréer l'image avec les couleurs sélectionnées
    selected_rgb = [pal[name] for name in st.session_state.selected_colors]
    new_img_arr = np.zeros_like(img_arr)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            lbl = labels[i * img_arr.shape[1] + j]
            new_img_arr[i, j] = selected_rgb[lbl % len(selected_rgb)]

    new_image = Image.fromarray(new_img_arr.astype("uint8"))
    st.image(new_image, caption="Image transformée", use_container_width=True)
