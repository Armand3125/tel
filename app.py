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
    "GA": (166, 169, 170), "VB": (94, 67, 183),
    "BF": (4, 47, 86),
}

# Couleurs fixes pour les clusters
fixed_colors = {
    0: "NC",  # Noir
    1: "VB",  # Violet basic
    2: "OM",  # Orange mandarine
    3: "BJ"   # Blanc jade
}

# Listes de palettes fixes
palettes = [
    ["NC", "RE", "JO", "BJ"],
    ["NC", "BF", "BG", "BJ"],
    ["NC", "BM", "BC", "JO"],
    ["NC", "VB", "VL", "BJ"]
]

st.title("Tylice Simplifié")

# Téléchargement de l'image
uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

# Sélection du nombre de couleurs
if "num_selections" not in st.session_state:
    st.session_state.num_selections = 4

# Limiter le nombre de sélections aux couleurs disponibles dans fixed_colors
num_selections = min(st.session_state.num_selections, len(fixed_colors))

# Traitement de l'image téléchargée
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    width, height = image.size
    dim = 300  # Image affichée plus petite
    new_width = dim if width > height else int((dim / height) * width)
    new_height = dim if height >= width else int((dim / width) * height)

    resized_image = image.resize((new_width, new_height))
    img_arr = np.array(resized_image)

    # Conversion de pixels à centimètres (350px = 14cm, soit 25px/cm)
    px_per_cm = 25
    new_width_cm = round(new_width / px_per_cm, 1)  # Arrondi à 1 décimale (en cm)
    new_height_cm = round(new_height / px_per_cm, 1)  # Arrondi à 1 décimale (en cm)

    if img_arr.shape[-1] == 3:
        pixels = img_arr.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_selections, random_state=0).fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Calculer les niveaux de gris pour chaque cluster
        grayscale_values = np.dot(centers, [0.2989, 0.5870, 0.1140])
        sorted_indices = np.argsort(grayscale_values)  # Trier par niveaux de gris

        # Associer les couleurs fixes aux clusters
        selected_colors = [pal[fixed_colors[i]] for i in range(num_selections)]

        new_img_arr = np.zeros_like(img_arr)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                lbl = labels[i * img_arr.shape[1] + j]
                new_color_index = np.where(sorted_indices == lbl)[0][0]
                new_img_arr[i, j] = selected_colors[new_color_index]

        new_image = Image.fromarray(new_img_arr.astype('uint8'))

        # Affichage de l'image transformée
        st.image(new_image, caption="Image transformée", use_container_width=False, width=dim)

        # Affichage des palettes fixes
        for palette in palettes:
            palette_colors = [pal[color] for color in palette]
            palette_image = np.zeros((50, 200, 3), dtype=np.uint8)
            section_width = 200 // len(palette_colors)

            for idx, color in enumerate(palette_colors):
                palette_image[:, idx * section_width:(idx + 1) * section_width] = color

            st.image(palette_image, caption=f"Palette: {' - '.join(palette)}", use_container_width=False, width=200)

# Affichage des conseils d'utilisation
st.markdown("""
    ### 📝 Conseils d'utilisation :
    - Les couleurs les plus compatibles avec l'image apparaissent en premier.
    - Préférez des images avec un bon contraste et des éléments bien définis.
    - Une **image carrée** donnera un meilleur résultat.
""", unsafe_allow_html=True)
