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
    "Or": (255, 204, 0), "Cyan": (0, 204, 255),
    "Lila": (204, 153, 255), "Vert": (51, 204, 51),
    "Rouge": (255, 0, 0), "Bleu": (0, 102, 204),
    "Orange": (255, 165, 0), "Vert foncé": (34, 139, 34),
    "Bleu clair": (135, 206, 235), "Magenta": (255, 0, 255),
    "Argent": (192, 192, 192), "Violet": (128, 0, 128),
    "Bleu foncé": (0, 0, 128),
}

# Configuration du style CSS amélioré
css = """
    <style>
        /* Styles généraux */
        .stApp {
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            font-family: 'Arial', sans-serif;
            background-color: #f0f0f0;
        }

        /* Titre */
        h1 {
            font-size: 3rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        }

        /* Fond de la page */
        .stMarkdown {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        /* Boutons de sélection des couleurs */
        .stButton {
            background-color: #007BFF;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.2s ease-in-out;
        }
        .stButton:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
        }

        /* Sélecteurs de couleurs */
        .stSelectbox {
            background-color: #ffffff;
            border-radius: 5px;
            border: 1px solid #ddd;
            margin-bottom: 15px;
            padding: 10px;
        }

        /* Alignement des éléments */
        .stImage, .stSelectbox, .stButton, .stFileUploader {
            margin-top: 20px;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        /* Style des cases de couleurs */
        .color-box {
            border: 2px solid #ddd;
            margin: 5px;
            width: 50px;
            height: 50px;
            display: inline-block;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease-in-out;
        }

        .color-box:hover {
            transform: scale(1.1);
        }

        /* Style des conseils */
        #conseils .stMarkdown {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            color: #333;
        }

        #conseils .stMarkdown h3 {
            font-size: 1.5rem;
            margin-bottom: 10px;
        }

        #conseils .stMarkdown p {
            font-size: 1rem;
            line-height: 1.6;
            color: #555;
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
            color_name = st.selectbox(f"Couleur dominante {i+1}", sorted_ordered_colors_by_cluster[i], key=f"color_select_{i}", index=0)
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

        # Encadré pour l'image
        st.markdown('<div style="border: 2px solid #ddd; border-radius: 8px; padding: 15px; margin-top: 20px;">', unsafe_allow_html=True)
        st.image(resized_image, caption="Aperçu de l'image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Convertir l'image générée en Base64
        img_base64 = encode_image_base64(new_image)

        # Afficher les dimensions
        st.markdown(f"Dimensions de l'image : {new_width} x {new_height}")

        # Bouton de téléchargement avec un fond coloré et une bordure douce
        st.download_button(
            label="Télécharger l'image",
            data=img_base64,
            file_name=file_name,
            mime="image/png",
            key="download_button",
            help="Cliquez pour télécharger votre image générée.",
            use_container_width=True
        )

    else:
        st.error("L'image doit être en RGB (3 canaux) pour continuer.")

# Section des conseils (avec un ID unique)
st.markdown('<div id="conseils">', unsafe_allow_html=True)
st.markdown("""
    ### 📝 Conseils d'utilisation :
    - 🖼️ Les couleurs les plus compatibles avec l'image apparaissent en premier.
    - 🌟 Préférez des images avec un bon contraste et des éléments bien définis.
    - 🔲 Une **image carrée** donnera un meilleur résultat.
    - ⚪ Il est recommandé d'inclure au moins une **zone de noir ou de blanc** pour assurer un bon contraste.
    - 🎨 Utiliser des **familles de couleurs** (ex: blanc, jaune, orange, rouge) peut produire des résultats visuellement intéressants.
    - 🌈 **Expérimentez** avec différentes combinaisons pour trouver l'esthétique qui correspond le mieux à votre projet !
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
