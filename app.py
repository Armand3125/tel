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
    ["NC", "VGa", "BM", "GA", "JO", "BJ"],  # Palette improvisée 1
    ["NC", "BF", "VGa", "VG", "VL", "BJ"],  # Palette improvisée 2
]

st.title("Tylice Simplifié")

# Téléchargement de l'image
uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

# Boutons pour sélectionner le mode
if "mode" not in st.session_state:
    st.session_state.mode = "4"

if uploaded_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("4 Couleurs : 7.95 €"):
            st.session_state.mode = "4"
    with col2:
        if st.button("6 Couleurs : 12.95 €"):
            st.session_state.mode = "6"

# Traitement de l'image téléchargée
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    width, height = image.size
    dim = 350  # Réduction à 350 pixels pour la plus grande dimension
    new_width = dim if width > height else int((dim / height) * width)
    new_height = dim if height >= width else int((dim / width) * height)

    resized_image = image.resize((new_width, new_height))
    img_arr = np.array(resized_image)

    if st.session_state.mode == "4":
        # Trouver 4 clusters avec KMeans
        pixels = img_arr.reshape(-1, 3)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Calculer les niveaux de gris des clusters
        grayscale_values = np.dot(centers, [0.2989, 0.5870, 0.1140])
        sorted_indices = np.argsort(grayscale_values)  # Trier du plus sombre au plus clair

        # Affichage de l'image recolorée pour chaque palette (2 par ligne)
        col_count = 0
        cols = st.columns(2)

        for palette in palettes:
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

    elif st.session_state.mode == "6":
        # Trouver 6 clusters avec KMeans
        pixels = img_arr.reshape(-1, 3)
        kmeans_6 = KMeans(n_clusters=6, random_state=0).fit(pixels)
        labels_6 = kmeans_6.labels_
        centers_6 = kmeans_6.cluster_centers_

        # Calculer les niveaux de gris des clusters pour 6 couleurs
        grayscale_values_6 = np.dot(centers_6, [0.2989, 0.5870, 0.1140])
        sorted_indices_6 = np.argsort(grayscale_values_6)  # Trier du plus sombre au plus clair

        # Affichage de l'image recolorée pour chaque palette à 6 couleurs
        col_count = 0
        cols = st.columns(2)

        for palette in palettes_6:
            palette_colors = [pal[color] for color in palette]

            recolored_img_arr = np.zeros_like(img_arr)
            for i in range(img_arr.shape[0]):
                for j in range(img_arr.shape[1]):
                    lbl = labels_6[i * img_arr.shape[1] + j]
                    sorted_index = np.where(sorted_indices_6 == lbl)[0][0]
                    recolored_img_arr[i, j] = palette_colors[sorted_index]

            recolored_image = Image.fromarray(recolored_img_arr.astype('uint8'))

            with cols[col_count % 2]:
                st.image(recolored_image, caption=f"Palette: {' - '.join(palette)}", use_container_width=False, width=dim)
            col_count += 1
