import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

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

# Configuration
st.title("Tylice - Sélection des Couleurs")
uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

# Choix du nombre de clusters
if "num_selections" not in st.session_state:
    st.session_state.num_selections = 4

if st.button("4 Couleurs : 7.95 €"):
    st.session_state.num_selections = 4
if st.button("6 Couleurs : 11.95 €"):
    st.session_state.num_selections = 6

num_selections = st.session_state.num_selections

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    img_arr = np.array(image)
    pixels = img_arr.reshape(-1, 3)

    # Appliquer KMeans pour extraire les couleurs dominantes
    kmeans = KMeans(n_clusters=num_selections, random_state=0).fit(pixels)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Trouver les couleurs les plus proches dans la palette
    pal_rgb = np.array(list(pal.values()))
    closest_colors = []
    for center in centers:
        distances = distance.cdist([center], pal_rgb, "euclidean")
        closest_color_idx = np.argmin(distances)
        closest_colors.append(list(pal.keys())[closest_color_idx])

    # Afficher les couleurs suggérées
    st.markdown("### Couleurs suggérées :")
    cols = st.columns(len(closest_colors))
    selected_colors = []
    for i, color_name in enumerate(closest_colors):
        color_rgb = pal[color_name]
        with cols[i]:
            st.markdown(
                f'<div style="width: 50px; height: 50px; background-color: rgb{color_rgb};"></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Choisir {color_name}", key=f"suggested_color_{i}"):
                selected_colors.append(color_rgb)

    # Compléter avec des couleurs par défaut si nécessaire
    while len(selected_colors) < num_selections:
        selected_colors.append((255, 255, 255))

    # Transformation de l'image
    cluster_to_color = {i: selected_colors[i] for i in range(num_selections)}
    new_img_arr = np.zeros_like(img_arr)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            cluster = labels[i * img_arr.shape[1] + j]
            new_img_arr[i, j] = cluster_to_color[cluster]

    # Afficher l'image transformée
    new_image = Image.fromarray(new_img_arr.astype("uint8"))
    st.image(new_image, caption="Image transformée", use_container_width=True)

    # Téléchargement de l'image
    st.download_button(
        "Télécharger l'image modifiée",
        data=new_image.tobytes(),
        file_name="image_transformee.png",
        mime="image/png",
    )
