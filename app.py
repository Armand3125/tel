import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
from datetime import datetime
import base64

# Palette de couleurs
pal = {
    "NC": (0, 0, 0), "BJ": (255, 255, 255),
    "JO": (228, 189, 104), "BC": (0, 134, 214),
    "VL": (174, 150, 212), "VG": (63, 142, 67),
    "RE": (222, 67, 67), "BM": (0, 120, 191),
    "OM": (249, 153, 99), "VGa": (59, 102, 94),
    "BG": (163, 216, 225), "VM": (236, 0, 140),
    "GA": (166, 169, 170), "VB": (94, 67, 183),
    "BF": (4, 47, 86),
}

# Configuration du style CSS simplifié
css = """
    <style>
        .color-box { border: 2px solid black; margin: 5px; width: 50px; height: 50px; display: inline-block; border-radius: 10px; }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Titre de l'application
st.title("Tylice - Sélection des Couleurs")

# Chargement de l'image
uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

# Sélection du nombre de couleurs
num_selections = st.radio("Choisissez le nombre de couleurs:", [4, 6])

# Fonction pour convertir l'image en Base64
def encode_image_base64(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

if uploaded_image is not None:
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
        
        # Affichage simplifié des couleurs
        st.markdown("Sélectionnez les couleurs :")
        for i, cluster_index in enumerate(sorted_indices):
            color_name = st.selectbox(f"Couleur dominante {i+1}", sorted_ordered_colors_by_cluster[i], key=f"color_select_{i}")
            selected_colors.append(pal[color_name])
            selected_color_names.append(color_name)

        # Recréer l'image avec les nouvelles couleurs
        new_img_arr = np.zeros_like(img_arr)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                lbl = labels[i * img_arr.shape[1] + j]
                new_color_index = np.where(sorted_indices == lbl)[0][0]
                new_img_arr[i, j] = selected_colors[new_color_index]

        new_image = Image.fromarray(new_img_arr.astype('uint8'))
        resized_image = new_image

        # Calculer le nom du fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{''.join(selected_color_names)}_{timestamp}.png"

        st.image(resized_image, caption="Aperçu de l'image", use_container_width=True)

        # Convertir l'image générée en Base64
        img_base64 = encode_image_base64(new_image)

        # Afficher les dimensions
        st.markdown(f"Dimensions de l'image : {new_width/25} cm x {new_height/25} cm")

        # Bouton de téléchargement
        st.download_button(
            label="Télécharger l'image",
            data=img_base64,
            file_name=file_name,
            mime="image/png"
        )

    else:
        st.error("L'image doit être en RGB (3 canaux) pour continuer.")
