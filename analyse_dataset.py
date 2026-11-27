from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

chemin = Path(input("Chemin du dossier du dataset : ").strip().strip('"'))
dossier_images = chemin / "images"
dossier_annotations = chemin / "annotations"

if not dossier_images.is_dir() or not dossier_annotations.is_dir():
    print("Erreur : le dossier doit contenir images et annotations.")
    raise SystemExit

images = [f for f in dossier_images.iterdir()
          if f.suffix.lower() in (".jpg", ".jpeg")]
fichiers_xml = list(dossier_annotations.glob("*.xml"))

classes = Counter()
dimensions = Counter()
nombre_avions = []

for fichier in fichiers_xml:
    racine = ET.parse(fichier).getroot()

    taille = racine.find("size")
    dimensions[(taille.findtext("width"), taille.findtext("height"))] += 1

    objets = racine.findall("object")
    nombre_avions.append(len(objets))

    for objet in objets:
        classe = objet.findtext("name")
        if classe:
            classes[classe] += 1

print("\nImages :", len(images))
print("Annotations :", len(fichiers_xml))
print("Avions annotés :", sum(classes.values()))
print("Classes :", len(classes))

if nombre_avions:
    print("Avions par image — minimum :", min(nombre_avions),
          "| maximum :", max(nombre_avions))

print("\nAvions par classe :")
for classe, nombre in sorted(classes.items()):
    print(classe, ":", nombre)

print("\nDimensions les plus fréquentes :")
for taille, nombre in dimensions.most_common(5):
    print(taille[0], "x", taille[1], "pixels :", nombre, "images")