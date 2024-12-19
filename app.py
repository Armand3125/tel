import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
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

# Configuration du style CSS pour les boîtes colorées
css = """
    <style>
        .color-box {
            display: inline-block;
            margin: 10px;
            width: 50px;
            height: 50px;
            border-radius: 5px;
            cursor: pointer;
        }
        .color-box:hover {
            border: 3px solid black;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Titre de l'application
st.title("Tylice - Sélection des Couleurs")

# Chargement de l'image
uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

# Sélection du nombre de couleurs avec les boutons
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

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    img_arr = np.array(image)
    pixels = img_arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_selections, random_state=0).fit(pixels)
    centers = kmeans.cluster_centers_

    centers_rgb = np.array(centers, dtype=int)
    pal_rgb = np.array(list(pal.values()), dtype=int)
    distances = np.linalg.norm(centers_rgb[:, None] - pal_rgb[None, :], axis=2)

    # Trouver les couleurs les plus proches dans la palette
    ordered_colors_by_cluster = []
    for i in range(num_selections):
        closest_colors_idx = distances[i].argsort()
        ordered_colors_by_cluster.append([list(pal.keys())[idx] for idx in closest_colors_idx])

    # Afficher les couleurs détectées sous forme de boîtes cliquables
    st.markdown("### Cliquez pour sélectionner les couleurs :")
    for i, cluster in enumerate(ordered_colors_by_cluster):
        st.markdown(f"**Couleur dominante {i+1}:**")
        cols = st.columns(len(cluster))
        for j, color_name in enumerate(cluster):
            with cols[j]:
                color = pal[color_name]
                color_hex = rgb_to_hex(color)
                if st.button("", key=f"color_button_{i}_{j}", help=color_name):
                    if len(st.session_state.selected_colors) < num_selections:
                        st.session_state.selected_colors.append((color_name, color))

    # Afficher les couleurs sélectionnées
    if st.session_state.selected_colors:
        st.markdown("### Couleurs sélectionnées :")
        for color_name, color in st.session_state.selected_colors:
            st.markdown(
                f'<div class="color-box" style="background-color: {rgb_to_hex(color)};"></div>',
                unsafe_allow_html=True,
            )

    # Recréer l'image avec les couleurs sélectionnées
    if len(st.session_state.selected_colors) == num_selections:
        selected_rgb = [color[1] for color in st.session_state.selected_colors]
        new_img_arr = np.zeros_like(img_arr)
        labels = kmeans.labels_
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                lbl = labels[i * img_arr.shape[1] + j]
                new_img_arr[i, j] = selected_rgb[lbl]

        new_image = Image.fromarray(new_img_arr.astype("uint8"))

        st.image(new_image, caption="Image transformée", use_container_width=True)
