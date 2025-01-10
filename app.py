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

# Traitement de l'image téléchargée
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    width, height = image.size
    dim = 300  # Image affichée plus petite
    new_width = dim if width > height else int((dim / height) * width)
    new_height = dim if height >= width else int((dim / width) * height)

    resized_image = image.resize((new_width, new_height))
    img_arr = np.array(resized_image)

    # Affichage de l'image recolorée pour chaque palette
    for palette in palettes:
        palette_colors = [pal[color] for color in palette]

        recolored_img_arr = np.zeros_like(img_arr)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                pixel = img_arr[i, j]
                distances = [np.linalg.norm(pixel - np.array(color)) for color in palette_colors]
                closest_color_index = np.argmin(distances)
                recolored_img_arr[i, j] = palette_colors[closest_color_index]

        recolored_image = Image.fromarray(recolored_img_arr.astype('uint8'))
        st.image(recolored_image, caption=f"Palette: {' - '.join(palette)}", use_container_width=False, width=dim)
