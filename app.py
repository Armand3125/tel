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

# Configuration du style CSS simplifié
css = """
    <style>
        .stApp {
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        .stButton, .stSelectbox, .stFileUploader, .stDownloadButton {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        .stImage {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        .color-box {
            border: 2px solid black;
            margin: 5px;
            width: 50px;
            height: 50px;
            display: inline-block;
            border-radius: 10px;
            text-align: center;
        }
    </style>
"""

# Intégrer le script JavaScript pour éviter l'affichage du clavier
js = """
    <script>
        window.addEventListener('focus', function(event) {
            document.activeElement.blur(); // Retirer le focus des éléments de saisie
        });
    </script>
"""

# Application de la configuration CSS et du script JavaScript
st.markdown(css, unsafe_allow_html=True)
st.markdown(js, unsafe_allow_html=True)

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

        # Section de sélection des couleurs (avec un ID unique)
        st.markdown('<div id="selection-couleurs">', unsafe_allow_html=True)
        st.markdown("Sélectionnez les couleurs :")
        for i, cluster_index in enumerate(sorted_indices):
            color_name = st.radio(f"Couleur dominante {i+1}", sorted_ordered_colors_by_cluster[i], key=f"color_select_{i}")
            selected_colors.append(pal[color_name])
            selected_color_names.append(color_name)
        st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown(f"Dimensions de l'image : {new_width} x {new_height}")

        # Bouton de téléchargement
        st.download_button(
            label="Télécharger l'image",
            data=img_base64,
            file_name=file_name,
            mime="image/png"
        )

    else:
        st.error("L'image doit être en RGB (3 canaux) pour continuer.")
