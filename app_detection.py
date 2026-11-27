import base64
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision import models, transforms
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor

from fichiers_techniques.detection_avions import construire_modele


CHECKPOINT = Path(__file__).resolve().parent / "fichiers_techniques" / "detecteur_8_epoques.pt"
CLASSIFIER_CHECKPOINT = Path(__file__).resolve().parent / "fichiers_techniques" / "modele_avions.pt"
HERO_IMAGES = [
    Path(__file__).resolve().parent / "images_banniere" / f"avion_{i}.jpg"
    for i in range(1, 5)
]
COULEURS = [
    "#ff4b4b", "#00a6fb", "#f7b801", "#2ec4b6", "#9b5de5",
    "#f15bb5", "#70e000", "#ff7f11", "#3a86ff", "#8338ec",
]


def selectionner_detections(boites, scores, seuil):
    candidats = (scores >= seuil).nonzero().flatten()
    # Une même boîte peut recevoir plusieurs classes ; préserver les avions proches.
    gardes = nms(boites[candidats], scores[candidats], 0.6)
    return candidats[gardes].tolist()


@st.cache_resource
def charger_detecteur():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {CHECKPOINT}")

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sauvegarde = torch.load(CHECKPOINT, map_location=appareil, weights_only=True)
    classes = sauvegarde["classes"]
    modele = construire_modele(preentraine=False)
    modele.load_state_dict(sauvegarde["modele"])
    modele.to(appareil)
    modele.eval()
    return modele, classes, appareil, sauvegarde["epoque"]


@st.cache_resource
def charger_classifieur():
    if not CLASSIFIER_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Classifieur introuvable : {CLASSIFIER_CHECKPOINT}")

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sauvegarde = torch.load(
        CLASSIFIER_CHECKPOINT, map_location=appareil, weights_only=True
    )
    classes = sauvegarde["classes"]
    modele = models.resnet50(weights=None)
    modele.fc = nn.Linear(modele.fc.in_features, len(classes))
    modele.load_state_dict(sauvegarde["poids"])
    modele.to(appareil)
    modele.eval()
    return modele, classes, appareil


PREPARER_CROP = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


st.set_page_config(page_title="AeroScope | Détection", page_icon="✈️", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #071116; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stFileUploader"] section {
        border: 1px dashed rgba(118,247,197,.42);
        border-radius: 12px;
        background: rgba(14,34,39,.7);
    }
    .st-key-choixmode [data-testid="stRadioOption"] {
        border: 1px solid rgba(118,247,197,.22);
        border-radius: 10px;
        padding: .5rem .85rem;
        background: #0e2227;
    }
    .st-key-choixmode [data-testid="stRadioOption"][data-selected="true"] {
        border-color: #76f7c5;
        background: #163a35;
    }
    [data-testid="stMetric"] {
        border-color: rgba(118,247,197,.25);
        border-radius: 12px;
        background: #0e2227;
    }
    [data-testid="stTable"] {
        border: 1px solid rgba(118,247,197,.25);
        border-radius: 12px;
        overflow: hidden;
    }
    .st-key-photoresultat [data-testid="stImage"] img {
        max-height: 55vh;
        max-width: 100%;
        width: auto;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .page-title {
        padding: .8rem 1.25rem;
        margin-bottom: .5rem;
        border: 1px solid rgba(118, 247, 197, .25);
        border-left: 4px solid #76f7c5;
        border-radius: 14px;
        background: #0e2227;
    }
    .page-title h1 {
        margin: 0;
        padding: 0;
        color: white;
        font-size: clamp(1.5rem, 3vw, 2.4rem);
        text-align: center;
        text-transform: uppercase;
    }
    </style>
    <header class="page-title"><h1>Détection d'avions</h1></header>
    """,
    unsafe_allow_html=True,
)

if all(chemin.is_file() for chemin in HERO_IMAGES):
    photos = [
        base64.b64encode(chemin.read_bytes()).decode("ascii")
        for chemin in HERO_IMAGES
    ]
    st.markdown(
        f"""
        <style>
        .hero {{
            position: relative;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            height: 210px;
            margin-bottom: .5rem;
            border: 1px solid rgba(118, 247, 197, .25);
            border-radius: 14px;
            overflow: hidden;
        }}
        .hero img {{
            width: 100%;
            height: 100%;
            min-height: 0;
            object-fit: cover;
        }}
        .hero img:nth-child(1) {{ object-position: center 54%; }}
        .hero img:nth-child(2) {{ object-position: 55% 60%; }}
        .hero img:nth-child(3) {{ object-position: 25% 60%; }}
        .hero img:nth-child(4) {{ object-position: center; }}
        .hero-description {{
            margin: 0 0 .5rem;
            color: #d7eee5;
            font-size: .95rem;
        }}
        @media (max-width: 600px) {{
            .hero {{ grid-template-columns: repeat(2, minmax(0, 1fr)); height: 240px; }}
        }}
        </style>
        <div class="hero" aria-label="Quatre avions en vol">
            <img src="data:image/jpeg;base64,{photos[0]}" alt="">
            <img src="data:image/jpeg;base64,{photos[1]}" alt="">
            <img src="data:image/jpeg;base64,{photos[2]}" alt="">
            <img src="data:image/jpeg;base64,{photos[3]}" alt="">
        </div>
        <p class="hero-description">Identifiez un avion ou repérez les appareils dans une scène complète.</p>
        """,
        unsafe_allow_html=True,
    )
else:
    st.error("Les photos de la bannière sont introuvables dans le dossier du projet.")

seuil = 0.5
with st.container(key="choixmode"):
    mode = st.radio(
        "Mode",
        ["Crop", "Scène complète"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode == "Crop":
        st.caption("Crop : identifiez le modèle d'un avion déjà cadré.")
    else:
        st.caption("Scène complète : détectez et identifiez les avions dans l'image.")
fichier = st.file_uploader(
    "Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if fichier is not None:
    try:
        with Image.open(fichier) as source:
            image = source.convert("RGB")
    except (OSError, ValueError) as erreur:
        st.error(f"Impossible d'ouvrir l'image : {erreur}")
        st.stop()

    if mode == "Crop":
        try:
            modele, classes, appareil = charger_classifieur()
        except (OSError, RuntimeError, KeyError, ValueError) as erreur:
            st.error(f"Impossible de charger le classifieur : {erreur}")
            st.stop()

        entree = PREPARER_CROP(image).unsqueeze(0).to(appareil)
        with torch.inference_mode():
            probabilites = torch.softmax(modele(entree), dim=1)[0].cpu()
        valeurs, indices = torch.topk(probabilites, k=min(3, len(classes)))
        st.markdown("#### Résultat")
        with st.container(key="photoresultat"):
            st.image(image, width="stretch")
        st.metric("Classe prédite", classes[indices[0].item()], border=True)
        st.table([
            {
                "Classe": classes[indice.item()],
                "Confiance": f"{valeur.item() * 100:.1f} %",
            }
            for valeur, indice in zip(valeurs, indices)
        ])
    else:
        try:
            modele, classes, appareil, epoques = charger_detecteur()
        except (OSError, RuntimeError, KeyError, ValueError) as erreur:
            st.error(f"Impossible de charger le détecteur : {erreur}")
            st.stop()

        with torch.inference_mode():
            sortie = modele([to_tensor(image).to(appareil)])[0]

        boites = sortie["boxes"].detach().cpu()
        etiquettes = sortie["labels"].detach().cpu()
        scores = sortie["scores"].detach().cpu()
        indices = selectionner_detections(boites, scores, seuil)

        annotee = image.copy()
        dessin = ImageDraw.Draw(annotee)
        for i in indices:
            x1, y1, x2, y2 = boites[i].tolist()
            etiquette = int(etiquettes[i])
            nom_classe = classes[etiquette]
            couleur = COULEURS[(etiquette - 1) % len(COULEURS)]
            dessin.rectangle((x1, y1, x2, y2), outline=couleur, width=3)
            texte = f"{nom_classe} {scores[i].item():.0%}"
            boite_texte = dessin.textbbox((x1, y1), texte)
            dessin.rectangle(boite_texte, fill=couleur)
            dessin.text((x1, y1), texte, fill="white")

        st.markdown("#### Résultat")
        st.metric("Avions détectés", len(indices), border=True)
        with st.container(key="photoresultat"):
            st.image(annotee, width="stretch")
        if indices:
            st.table([
                {
                    "Classe": classes[int(etiquettes[i])],
                    "Confiance": f"{scores[i].item() * 100:.1f} %",
                }
                for i in indices
            ])
