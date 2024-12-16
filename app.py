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

# Configuration du style CSS
css = """
    <style>
        .color-box { border: 3px solid black; }
        .first-box { margin-top: 15px; }
        .button-container { text-align: center; margin-bottom: 20px; }
        .color-container { text-align: center; margin-top: 5px; }
        .percentage-container { text-align: center; }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Titre de l'application
st.title("Tylice")

# Chargement de l'image
uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

# Sélection du nombre de couleurs
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

    # Conversion de pixels à centimètres
    px_per_cm = 25
    new_width_cm = round(new_width / px_per_cm, 1)  # Arrondi à 1 décimale (en cm)
    new_height_cm = round(new_height / px_per_cm, 1)  # Arrondi à 1 décimale (en cm)

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
        for i, cluster_index in enumerate(sorted_indices):
            st.markdown(f"<div class='color-container'>", unsafe_allow_html=True)
            for j, color_name in enumerate(sorted_ordered_colors_by_cluster[i]):
                color_rgb = pal[color_name]
                margin_class = "first-box" if j == 0 else ""
                st.markdown(
                    f"<div class='color-box {margin_class}' style='background-color: rgb{color_rgb}; width: 80px; height: 20px; border-radius: 5px; margin-bottom: 4px;'></div>",
                    unsafe_allow_html=True
                )
            st.markdown(f"</div>", unsafe_allow_html=True)

            selected_color_name = st.radio("", sorted_ordered_colors_by_cluster[i], key=f"radio_{i}")
            selected_colors.append(pal[selected_color_name])
            selected_color_names.append(selected_color_name)

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

        st.image(resized_image, caption="Aperçu de l'image", use_column_width=True)

        # Convertir l'image générée en Base64
        img_base64 = encode_image_base64(new_image)

        st.markdown(f"**{new_width_cm} cm x {new_height_cm} cm**")

        st.download_button(
            label="Télécharger l'image",
            data=img_base64,
            file_name=file_name,
            mime="image/png"
        )

        # Script pour envoyer l'image à Wix via postMessage
        st.write(
            f"""
            <script>
            const data = {{
                name: "Image personnalisée",
                price: 19.99,
                fileData: "{img_base64}",
                fileName: "{file_name}"
            }};
            window.parent.postMessage(data, "https://www.tylice.com/");
            </script>
            """,
            unsafe_allow_html=True
        )

    else:
        st.error("L'image doit être en RGB (3 canaux) pour continuer.")

# Informations supplémentaires sur l'utilisation
st.markdown("""
    ### 📝 Conseils d'utilisation :
    - Les couleurs les plus compatibles avec l'image apparaissent en premier.
    - Préférez des images avec un bon contraste et des éléments bien définis.
    - Une **image carrée** donnera un meilleur résultat.
    - Il est recommandé d'inclure au moins une **zone de noir ou de blanc** pour assurer un bon contraste.
    - Utiliser des **familles de couleurs** (ex: blanc, jaune, orange, rouge) peut produire des résultats visuellement intéressants.
    - **Expérimentez** avec différentes combinaisons pour trouver l'esthétique qui correspond le mieux à votre projet !
""", unsafe_allow_html=True)
