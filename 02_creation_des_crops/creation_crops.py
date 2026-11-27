from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

# Demander le chemin du dossier principal du jeu de données
dossier_dataset = Path(input("Chemin du dossier du jeu de données : ").strip().strip('"'))

dossier_images = dossier_dataset / "images"
dossier_annotations = dossier_dataset / "annotations"
dossier_crops = dossier_dataset / "crops_par_classe"

taille_crop = 224
marge = 1.0  # 100 % de marge autour de l'avion

# Associer les noms des images à leurs chemins
images_par_nom = {}
extensions = {".jpg", ".jpeg", ".png"}

for chemin_image in dossier_images.iterdir():
    if chemin_image.is_file() and chemin_image.suffix.lower() in extensions:
        images_par_nom[chemin_image.stem.lower()] = chemin_image

nombre_crops = 0

# Lire chaque fichier XML
for chemin_xml in sorted(dossier_annotations.glob("*.xml")):
    nom_image = chemin_xml.stem.lower()
    chemin_image = images_par_nom.get(nom_image)

    if chemin_image is None:
        print(f"Image introuvable pour l'annotation : {chemin_xml.name}")
        continue

    image = Image.open(chemin_image).convert("RGB")
    largeur_image, hauteur_image = image.size
    racine = ET.parse(chemin_xml).getroot()

    # Parcourir les avions annotés dans l'image
    for numero, objet in enumerate(racine.findall("object"), start=1):
        classe = (objet.findtext("name") or "").strip()
        boite = objet.find("bndbox")

        if not classe or boite is None:
            continue

        xmin = int(boite.findtext("xmin"))
        ymin = int(boite.findtext("ymin"))
        xmax = int(boite.findtext("xmax"))
        ymax = int(boite.findtext("ymax"))

        # Pascal VOC commence ses coordonnées à 1
        xmin = max(0, xmin - 1)
        ymin = max(0, ymin - 1)
        xmax = min(largeur_image, xmax)
        ymax = min(hauteur_image, ymax)

        largeur_avion = xmax - xmin
        hauteur_avion = ymax - ymin

        if largeur_avion <= 0 or hauteur_avion <= 0:
            print(f"Boîte invalide dans {chemin_xml.name}, objet {numero}")
            continue

        # Construire un crop carré avec une marge autour de l'avion
        cote = int(max(largeur_avion, hauteur_avion) * (1 + 2 * marge))
        centre_x = (xmin + xmax) / 2
        centre_y = (ymin + ymax) / 2

        gauche = round(centre_x - cote / 2)
        haut = round(centre_y - cote / 2)
        droite = gauche + cote
        bas = haut + cote

        crop = image.crop((gauche, haut, droite, bas))
        crop = crop.resize((taille_crop, taille_crop), Image.Resampling.LANCZOS)

        # Ranger le crop dans un dossier correspondant à sa classe
        dossier_classe = dossier_crops / classe
        dossier_classe.mkdir(parents=True, exist_ok=True)

        nom_crop = f"{chemin_xml.stem}_{numero:02d}.png"
        crop.save(dossier_classe / nom_crop)
        nombre_crops += 1

print(f"Terminé : {nombre_crops} crops créés dans {dossier_crops}")