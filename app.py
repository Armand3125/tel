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

st.title("Tylice Combiné")

# Téléchargement de l'image
uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

# Choix entre propositions et personnalisation
if "mode" not in st.session_state:
    st.session_state.mode = "propositions"

if uploaded_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Propositions pré-faites"):
            st.session_state.mode = "propositions"
    with col2:
        if st.button("Personnalisation avancée"):
            st.session_state.mode = "personnalisation"

# Afficher les propositions
if uploaded_image is not None and st.session_state.mode == "propositions":
    image = Image.open(uploaded_image).convert("RGB")
    width, height = image.size
    dim = 350
    new_width = dim if width > height else int((dim / height) * width)
    new_height = dim if height >= width else int((dim / width) * height)

    resized_image = image.resize((new_width, new_height))
    img_arr = np.array(resized_image)

    pixels = img_arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    grayscale_values = np.dot(centers, [0.2989, 0.5870, 0.1140])
    sorted_indices = np.argsort(grayscale_values)

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

# Afficher la personnalisation
elif uploaded_image is not None and st.session_state.mode == "personnalisation":
    image = Image.open(uploaded_image).convert("RGB")
    width, height = image.size
    dim = 350
    new_width = dim if width > height else int((dim / height) * width)
    new_height = dim if height >= width else int((dim / width) * height)

    resized_image = image.resize((new_width, new_height))
    img_arr = np.array(resized_image)

    st.write("Choisissez le nombre de couleurs :")
    num_selections = st.radio("Nombre de couleurs", [4, 6], index=0, horizontal=True)

    kmeans = KMeans(n_clusters=num_selections, random_state=0).fit(img_arr.reshape(-1, 3))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    cluster_counts = np.bincount(labels)
    total_pixels = len(labels)
    cluster_percentages = (cluster_counts / total_pixels) * 100

    sorted_indices = np.argsort(-cluster_percentages)
    cols = st.columns(num_selections * 2)

    selected_colors = []
    for i, cluster_index in enumerate(sorted_indices):
        with cols[i * 2]:
            st.write(f"Cluster {i+1} : {cluster_percentages[cluster_index]:.2f}%")
        with cols[i * 2 + 1]:
            color_name = st.radio(
                f"Couleurs pour Cluster {i+1}",
                list(pal.keys()),
                index=0,
                key=f"color_{i}",
            )
            selected_colors.append(pal[color_name])

    recolored_img_arr = np.zeros_like(img_arr)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            lbl = labels[i * img_arr.shape[1] + j]
            recolored_img_arr[i, j] = selected_colors[lbl]

    recolored_image = Image.fromarray(recolored_img_arr.astype('uint8'))
    st.image(recolored_image, caption="Image Personnalisée", use_container_width=True)
