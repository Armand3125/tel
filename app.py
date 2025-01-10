import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
import requests
import urllib.parse

# =========================
# Dictionnaire des couleurs
# =========================
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

# ====================================
# Listes de palettes fixes pour les Exemples
# ====================================
palettes_examples_4 = [
    ["NC", "RE", "JO", "BJ"],
    ["NC", "BM", "BG", "BJ"],
    ["NC", "BM", "JO", "BJ"],
    ["NC", "VB", "OM", "BJ"],
]

palettes_examples_6 = [
    ["NC", "VB", "RE", "OM", "JO", "BJ"],
    ["NC", "BF", "BM", "BC", "BG", "BJ"],
    ["NC", "VGa", "BM", "GA", "JO", "BJ"],  # Palette improvisée 1
    ["NC", "BF", "VGa", "VG", "VL", "BJ"],  # Palette improvisée 2
]

# ====================================
# Configuration du titre et du style
# ====================================
st.title("Tylice")

css = """
    <style>
        .stRadio div [data-testid="stMarkdownContainer"] p { display: none; }
        .radio-container { display: flex; flex-direction: column; align-items: center; margin: 10px; }
        .color-container { display: flex; flex-direction: column; align-items: center; margin-top: 5px; }
        .color-box { border: 3px solid black; }
        .stColumn { padding: 0 !important; }
        .first-box { margin-top: 15px; }
        .percentage-container { margin-bottom: 0; }
        .button-container { margin-bottom: 20px; }
        .shopify-link { font-size: 20px; font-weight: bold; text-decoration: none; color: #2e86de; }
        .dimension-text { font-size: 16px; font-weight: bold; color: #555; }
        .add-to-cart-button { margin-top: 10px; }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# =========================================
# Section 1: Téléchargement et Sélection
# =========================================
# Téléchargement de l'image
uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

# Sélection du nombre de couleurs en haut de la page
col1_top, col2_top = st.columns([2, 5])

if uploaded_image is not None:
    with col1_top:
        if st.button("4 Couleurs : 7.95 €", key="select_4_top"):
            st.session_state.num_selections = 4
    with col2_top:
        if st.button("6 Couleurs : 11.95 €", key="select_6_top"):
            st.session_state.num_selections = 6
else:
    # Affichage des boutons même sans image
    with col1_top:
        if st.button("4 Couleurs : 7.95 €", key="select_4_top_no_image"):
            st.session_state.num_selections = 4
    with col2_top:
        if st.button("6 Couleurs : 11.95 €", key="select_6_top_no_image"):
            st.session_state.num_selections = 6

# Initialisation du nombre de sélections
if "num_selections" not in st.session_state:
    st.session_state.num_selections = 4  # Valeur par défaut

num_selections = st.session_state.num_selections

# =========================================
# Affichage des conseils d'utilisation
# =========================================
st.markdown("""
    ### 📝 Conseils d'utilisation :
    - Les couleurs les plus compatibles avec l'image apparaissent en premier.
    - Préférez des images avec un bon contraste et des éléments bien définis.
    - Une **image carrée** donnera un meilleur résultat.
    - Il est recommandé d'inclure au moins une **zone de noir ou de blanc** pour assurer un bon contraste.
    - Utiliser des **familles de couleurs** (ex: blanc, jaune, orange, rouge) peut produire des résultats visuellement intéressants.
    - **Expérimentez** avec différentes combinaisons pour trouver l'esthétique qui correspond le mieux à votre projet !
""", unsafe_allow_html=True)

# =========================================
# Sections conditionnelles après upload d'image
# =========================================
if uploaded_image is not None:
    # =========================================
    # Fonctionnalités Réutilisables
    # =========================================

    def upload_to_cloudinary(image_buffer):
        """
        Uploads an image to Cloudinary and returns the secure URL.
        """
        url = "https://api.cloudinary.com/v1_1/dprmsetgi/image/upload"
        files = {"file": image_buffer}
        data = {"upload_preset": "image_upload_tylice"}
        try:
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                return response.json()["secure_url"]
            else:
                st.error(f"Erreur Cloudinary: {response.text}")
                return None
        except Exception as e:
            st.error(f"Erreur Cloudinary: {e}")
            return None

    def generate_shopify_cart_url(cloudinary_url, num_colors):
        """
        Generates a Shopify cart URL with the given image URL and variant ID based on the number of colors.
        """
        variant_id = "50063717106003" if num_colors == 4 else "50063717138771"
        encoded_image_url = urllib.parse.quote(cloudinary_url)
        shopify_cart_url = (
            f"https://tylice2.myshopify.com/cart/add?id={variant_id}&quantity=1&properties[Image]={encoded_image_url}"
        )
        return shopify_cart_url

    def process_image(image, num_clusters):
        """
        Processes the image by resizing and applying KMeans clustering.
        Returns the resized image array, labels, and sorted cluster indices.
        """
        width, height = image.size
        dim = 350  # Réduction à 350 pixels pour la plus grande dimension
        new_width = dim if width > height else int((dim / height) * width)
        new_height = dim if height >= width else int((dim / width) * height)

        resized_image = image.resize((new_width, new_height))
        img_arr = np
