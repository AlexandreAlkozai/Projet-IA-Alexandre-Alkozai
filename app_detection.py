import base64
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision import models, transforms
from torchvision.transforms.functional import to_tensor

from detection_avions import construire_modele


DOSSIER = Path(__file__).resolve().parent
CHECKPOINT = DOSSIER / "detecteur_8_epoques.pt"
CLASSIFIER_CHECKPOINT = DOSSIER / "modele_avions.pt"
HERO_IMAGE = DOSSIER / "f14_nasa.jpg"
COULEURS = [
    "#ff4b4b", "#00a6fb", "#f7b801", "#2ec4b6", "#9b5de5",
    "#f15bb5", "#70e000", "#ff7f11", "#3a86ff", "#8338ec",
]


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
st.markdown('<style>.stApp{background:#071116}'
            '[data-testid="stFileUploader"] section{border:1px dashed '
            'rgba(118,247,197,.42);border-radius:12px;background:rgba(14,34,39,.7)}'
            '</style>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .page-title {
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
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

if HERO_IMAGE.is_file():
    photo = base64.b64encode(HERO_IMAGE.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <style>
        .hero {{
            min-height: 300px;
            display: flex;
            align-items: flex-end;
            box-sizing: border-box;
            padding: 2rem;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(118, 247, 197, .25);
            border-radius: 14px;
            background-image:
                linear-gradient(90deg, rgba(3, 10, 14, .78), rgba(3, 10, 14, .08)),
                url("data:image/jpeg;base64,{photo}");
            background-size: cover;
            background-position: center 48%;
        }}
        </style>
        <div class="hero"></div>
        """,
        unsafe_allow_html=True,
    )

seuil = 0.5
mode = st.radio(
    "Mode",
    ["Crop", "Scène complète"],
    horizontal=True,
    label_visibility="collapsed",
)
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
        st.image(image, width="stretch")
        st.metric("Classe prédite", classes[indices[0].item()])
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
        indices = sorted(
            (scores >= seuil).nonzero().flatten().tolist(),
            key=lambda i: scores[i].item(),
            reverse=True,
        )

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

        st.metric("Avions détectés", len(indices))
        st.image(annotee, width="stretch")
        if indices:
            st.table([
                {
                    "Classe": classes[int(etiquettes[i])],
                    "Confiance": f"{scores[i].item() * 100:.1f} %",
                }
                for i in indices
            ])
