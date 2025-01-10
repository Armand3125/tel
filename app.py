import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
from datetime import datetime
import requests
import urllib.parse

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
    ["NC", "VGa", "BM", "GA", "JO", "BJ"],  # Palette improvisée 1
    ["NC", "BF", "VGa", "VG", "VL", "BJ"],  # Palette improvisée 2
]

st.title("Tylice")

# Style personnalisé
css = """
    <style>
        .stRadio div [data-testid="stMarkdownContainer"] p { display: none; }
        .radio-container { display: flex; flex-direction: column; align-items: center; margin: 10px; }
        .color-container { display: flex; flex-direction: row; align-items: center; margin: 5px 0; }
        .color-box { border: 3px solid black; margin-right: 8px; }
        .stColumn { padding: 0 !important; }
        .first-box { margin-top: 15px; }
        .percentage-container { margin-bottom: 0; }
        .button-container { margin-bottom: 20px; }
        .shopify-link { font-size: 20px; font-weight: bold; text-decoration: none; color: #2e86de; }
        .dimension-text { font-size: 16px; font-weight: bold; color: #555; }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Téléchargement de l'image
uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

# Sélection du nombre de couleurs
if "num_selections" not in st.session_state:
    st.session_state.num_selections = 4
if "mode" not in st.session_state:
    st.session_state.mode = "predefined"

col1, col2 = st.columns([2, 5])

with col1:
    if st.button("4 Couleurs : 7.95 €"):
        st.session_state.num_selections = 4

with col2:
    if st.button("6 Couleurs : 11.95 €"):
        st.session_state.num_selections = 6

col3, col4 = st.columns([2, 5])
with col3:
    if st.button("Compositions pré-définies"):
        st.session_state.mode = "predefined"

with col4:
    if st.button("Personnalisation avancée"):
        st.session_state.mode = "custom"

num_selections = st.session_state.num_selections

# Mode compositions pré-définies
if uploaded_image is not None and st.session_state.mode == "predefined":
    image = Image.open(uploaded_image).convert("RGB")
    width, height = image.size
    dim = 350
    new_width = dim if width > height else int((dim / height) * width)
    new_height = dim if height >= width else int((dim / width) * height)

    resized_image = image.resize((new_width, new_height))
    img_arr = np.array(resized_image)

    pixels = img_arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=4 if num_selections == 4 else 6, random_state=0).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    grayscale_values = np.dot(centers, [0.2989, 0.5870, 0.1140])
    sorted_indices = np.argsort(grayscale_values)

    palettes_to_use = palettes if num_selections == 4 else palettes_6

    col_count = 0
    cols = st.columns(2)

    for palette in palettes_to_use:
        palette_colors = [pal[color] for color in palette]

        recolored_img_arr = np.zeros_like(img_arr)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                lbl = labels[i * img_arr.shape[1] + j]
                sorted_index = np.where(sorted_indices == lbl)[0][0]
                recolored_img_arr[i, j] = palette_colors[sorted_index]

        recolored_image = Image.fromarray(recolored_img_arr.astype('uint8'))

        with cols[col_count % 2]:
            st.image(recolored_image, caption=f"Palette: {' - '.join(palette)}", use_container_width=False, width=dim)
        col_count += 1

# Mode personnalisation avancée
elif uploaded_image is not None and st.session_state.mode == "custom":
    image = Image.open(uploaded_image).convert("RGB")
    width, height = image.size
    dim = 350
    new_width = dim if width > height else int((dim / height) * width)
    new_height = dim if height >= width else int((dim / width) * height)

    resized_image = image.resize((new_width, new_height))
    img_arr = np.array(resized_image)

    if img_arr.shape[-1] == 3:
        pixels = img_arr.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_selections, random_state=0).fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        centers_rgb = np.array(centers, dtype=int)
        pal_rgb = np.array(list(pal.values()), dtype=int)
        distances = np.linalg.norm(centers_rgb[:, None] - pal_rgb[None, :], axis=2)

        ordered_colors_by_cluster = []
        for i in range(num_selections):
            closest_colors_idx = distances[i].argsort()
            ordered_colors_by_cluster.append([list(pal.keys())[idx] for idx in closest_colors_idx])

        cluster_counts = np.bincount(labels)
        total_pixels = len(labels)
        cluster_percentages = (cluster_counts / total_pixels) * 100

        sorted_indices = np.argsort(-cluster_percentages)
        sorted_percentages = cluster_percentages[sorted_indices]
        sorted_ordered_colors_by_cluster = [ordered_colors_by_cluster[i] for i in sorted_indices]

        selected_colors = []
        selected_color_names = []
        cols = st.columns(num_selections)

        for i, cluster_index in enumerate(sorted_indices):
            with cols[i]:
                st.markdown("<div class='color-container'>", unsafe_allow_html=True)
                for color_name in sorted_ordered_colors_by_cluster[i]:
                    color_rgb = pal[color_name]
                    st.markdown(
                        f"<div class='color-box' style='background-color: rgb{color_rgb}; width: 80px; height: 20px; margin: 4px 0; border-radius: 5px; display: inline-block;'></div>",
                        unsafe_allow_html=True
                    )
                st.radio("", sorted_ordered_colors_by_cluster[i], key=f"radio_{i}", label_visibility="hidden")

        recolored_img_arr = np.zeros_like(img_arr)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                lbl = labels[i * img_arr.shape[1] + j]
                new_color_index = np.where(sorted_indices == lbl)[0][0]
                recolored_img_arr[i, j] = pal[sorted_ordered_colors_by_cluster[new_color_index][0]]

        new_image = Image.fromarray(recolored_img_arr.astype('uint8'))
        st.image(new_image, caption="Image personnalisée", use_container_width=True)
