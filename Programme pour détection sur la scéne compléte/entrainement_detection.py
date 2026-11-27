from pathlib import Path
import subprocess
import sys
import tempfile

images = Path(input("Chemin du dossier images : ").strip().strip('"'))
annotations = Path(input("Chemin du dossier annotations : ").strip().strip('"'))

if not images.is_dir() or images.name.lower() != "images":
    raise ValueError("Le chemin doit désigner le dossier images.")

if not annotations.is_dir() or annotations.name.lower() != "annotations":
    raise ValueError("Le chemin doit désigner le dossier annotations.")

if images.resolve().parent != annotations.resolve().parent:
    raise ValueError("Les dossiers images et annotations doivent avoir le même parent.")

racine = Path(__file__).resolve().parent.parent
script = racine / "fichiers_techniques" / "detection_avions.py"

if not script.is_file():
    raise FileNotFoundError(f"Fichier introuvable : {script}")

resultats = Path(tempfile.mkdtemp(
    prefix="test_detection_",
    dir=racine
))
modele = resultats / "detecteur_8_epoques.pt"

print(f"Nouveau modèle : {modele}", flush=True)

subprocess.run(
    [
        sys.executable, "-u", str(script),
        "--dataset", str(images.resolve().parent),
        "--epoques", "8",
        "--sortie", str(modele),
    ],
    check=True
)