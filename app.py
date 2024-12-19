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

# Initialiser la sélection des couleurs
if "selected_colors" not in st.session_state:
    st.session_state.selected_colors = []

# Fonction pour convertir une couleur en HTML
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

# Fonction pour ajouter une couleur à la sélection
def add_color_to_selection(color_name):
    if color_name in pal:  # Vérifie que la couleur existe dans la palette
        if len(st.session_state.selected_colors) < st.session_state.num_selections:
            if color_name not in st.session_state.selected_colors:
                st.session_state.selected_colors.append(color_name)

# Interface principale
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
cols = st.columns(6)
for i, (color_name, color_rgb) in enumerate(pal.items()):
    with cols[i % 6]:
        btn_clicked = st.button(f"Choisir {color_name}", key=f"color_button_{color_name}")
        if btn_clicked:
            add_color_to_selection(color_name)

# Afficher les couleurs sélectionnées
if st.session_state.selected_colors:
    st.markdown("### Couleurs sélectionnées :")
    selected_cols = st.columns(len(st.session_state.selected_colors))
    for i, color_name in enumerate(st.session_state.selected_colors):
        if color_name in pal:  # Vérifie que la couleur est encore valide
            color_rgb = pal[color_name]
            color_hex = rgb_to_hex(color_rgb)
            with selected_cols[i]:
                st.markdown(
                    f'<div style="width: 50px; height: 50px; background-color: {color_hex};"></div>',
                    unsafe_allow_html=True,
                )

# Traitement de l'image
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    img_arr = np.array(image)
    pixels = img_arr.reshape(-1, 3)

    # Appliquer KMeans pour trouver les clusters
    kmeans = KMeans(n_clusters=st.session_state.num_selections, random_state=0).fit(pixels)
    labels = kmeans.labels_

    # Générer un mappage entre les clusters et les couleurs sélectionnées
    selected_rgb = [pal[name] for name in st.session_state.selected_colors]
    if len(selected_rgb) < st.session_state.num_selections:
        # Remplir avec des couleurs par défaut si nécessaire
        selected_rgb.extend([(255, 255, 255)] * (st.session_state.num_selections - len(selected_rgb)))

    cluster_to_color = {cluster: selected_rgb[cluster % len(selected_rgb)] for cluster in range(st.session_state.num_selections)}

    # Transformer l'image
    new_img_arr = np.zeros_like(img_arr)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            cluster = labels[i * img_arr.shape[1] + j]
            new_img_arr[i, j] = cluster_to_color[cluster]

    new_image = Image.fromarray(new_img_arr.astype("uint8"))
    st.image(new_image, caption="Image transformée", use_container_width=True)
