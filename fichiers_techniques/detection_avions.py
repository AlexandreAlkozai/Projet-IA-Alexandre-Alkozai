import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_tensor


DATASET = Path(__file__).resolve().parent.parent / "dataset"
CLASSES = ["fond"] + [f"A{i}" for i in range(1, 21)]


class ImagesAvions(Dataset):
    def __init__(self, dossier):
        images = dossier / "images"
        annotations = dossier / "annotations"
        if not images.is_dir() or not annotations.is_dir():
            raise FileNotFoundError("Les dossiers images et annotations sont nécessaires.")

        fichiers_images = {p.stem: p for p in images.glob("*.jpg")}
        fichiers_xml = {p.stem: p for p in annotations.glob("*.xml")}
        if not fichiers_images or fichiers_images.keys() != fichiers_xml.keys():
            raise ValueError("Les images et annotations doivent avoir les mêmes noms.")
        self.paires = [
            (fichiers_images[nom], fichiers_xml[nom])
            for nom in sorted(fichiers_images)
        ]

    def __len__(self):
        return len(self.paires)

    def __getitem__(self, index):
        chemin_image, chemin_xml = self.paires[index]
        with Image.open(chemin_image) as source:
            image = source.convert("RGB")
        largeur, hauteur = image.size
        racine = ET.parse(chemin_xml).getroot()
        boites, etiquettes = [], []

        for objet in racine.findall("object"):
            classe = (objet.findtext("name") or "").strip()
            if classe not in CLASSES[1:]:
                raise ValueError(f"{chemin_xml.name} : classe inconnue {classe!r}")
            boite = objet.find("bndbox")
            if boite is None:
                raise ValueError(f"{chemin_xml.name} : boîte absente")
            x1 = float(boite.findtext("xmin")) - 1
            y1 = float(boite.findtext("ymin")) - 1
            x2 = float(boite.findtext("xmax"))
            y2 = float(boite.findtext("ymax"))
            if not (0 <= x1 < x2 <= largeur and 0 <= y1 < y2 <= hauteur):
                raise ValueError(f"{chemin_xml.name} : boîte invalide")
            boites.append([x1, y1, x2, y2])
            etiquettes.append(CLASSES.index(classe))

        cible = {
            "boxes": torch.tensor(boites, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(etiquettes, dtype=torch.int64),
        }
        return to_tensor(image), cible


def assembler(lot):
    return tuple(zip(*lot))


def construire_modele(preentraine=True):
    poids = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if preentraine else None
    modele = fasterrcnn_resnet50_fpn(
        weights=poids,
        weights_backbone=None,
        box_detections_per_img=100,
    )
    entrees = modele.roi_heads.box_predictor.cls_score.in_features
    modele.roi_heads.box_predictor = FastRCNNPredictor(entrees, len(CLASSES))
    return modele


@torch.inference_mode()
def evaluer(modele, chargeur, appareil, seuil=0.5):
    modele.eval()
    vrais_positifs = predictions = objets = 0
    for images, cibles in chargeur:
        sorties = modele([image.to(appareil) for image in images])
        for sortie, cible in zip(sorties, cibles):
            garder = sortie["scores"].cpu() >= seuil
            boites = sortie["boxes"].cpu()[garder]
            labels = sortie["labels"].cpu()[garder]
            scores = sortie["scores"].cpu()[garder]
            attendu = cible["boxes"]
            attendu_labels = cible["labels"]
            predictions += len(boites)
            objets += len(attendu)
            if len(boites) == 0 or len(attendu) == 0:
                continue
            ious = box_iou(boites, attendu)
            deja_associes = set()
            for i in scores.argsort(descending=True).tolist():
                candidats = [
                    j for j in range(len(attendu))
                    if j not in deja_associes
                    and labels[i] == attendu_labels[j]
                    and ious[i, j] >= 0.5
                ]
                if candidats:
                    meilleur = max(candidats, key=lambda j: ious[i, j].item())
                    deja_associes.add(meilleur)
                    vrais_positifs += 1
    precision = vrais_positifs / predictions if predictions else 0.0
    rappel = vrais_positifs / objets if objets else 0.0
    return precision, rappel


def main():
    parser = argparse.ArgumentParser(description="Entraîner le détecteur d'avions")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--epoques", type=int, required=True)
    parser.add_argument("--sortie", type=Path, default=Path("detecteur_avions.pt"))
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--reprendre", action="store_true")
    args = parser.parse_args()
    if args.epoques < 1:
        parser.error("--epoques doit être supérieur à zéro")

    dataset = ImagesAvions(args.dataset)
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    taille_validation = max(1, round(len(indices) * 0.2))
    if len(indices) <= taille_validation:
        raise ValueError("Il faut au moins deux images pour séparer les données.")
    val_indices = indices[:taille_validation]
    train_indices = indices[taille_validation:]
    train = DataLoader(Subset(dataset, train_indices), batch_size=1, shuffle=True,
                       collate_fn=assembler)
    val = DataLoader(Subset(dataset, val_indices), batch_size=1,
                     collate_fn=assembler)
    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {appareil} | entraînement : {len(train_indices)} images"
          f" | validation : {len(val_indices)} images", flush=True)

    modele = construire_modele(preentraine=not args.reprendre).to(appareil)
    optimiseur = torch.optim.SGD(modele.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=0.0005)
    debut = 0
    if args.reprendre:
        sauvegarde = torch.load(args.sortie, map_location=appareil, weights_only=True)
        if sauvegarde["classes"] != CLASSES:
            raise ValueError("Classes du modèle incompatibles.")
        modele.load_state_dict(sauvegarde["modele"])
        debut = sauvegarde["epoque"]
        print(f"Reprise après l'époque {debut} avec un optimiseur réinitialisé.",
              flush=True)
    if debut >= args.epoques:
        raise ValueError("Le nombre d'époques demandé doit dépasser l'époque sauvegardée.")

    for epoque in range(debut + 1, args.epoques + 1):
        modele.train()
        perte_totale = 0.0
        for numero, (images, cibles) in enumerate(train, 1):
            images = [image.to(appareil) for image in images]
            cibles = [{k: v.to(appareil) for k, v in cible.items()} for cible in cibles]
            pertes = modele(images, cibles)
            perte = sum(pertes.values())
            if not torch.isfinite(perte):
                raise RuntimeError(f"Perte non finie à l'image {numero} : {perte.item()}")
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()
            perte_totale += perte.item()
            if numero % 50 == 0 or numero == len(train):
                print(f"Époque {epoque}/{args.epoques} : {numero}/{len(train)} images",
                      flush=True)

        # Sauvegarder avant la validation pour ne pas perdre l'époque terminée.
        temporaire = args.sortie.with_suffix(args.sortie.suffix + ".tmp")
        torch.save({
            "epoque": epoque,
            "modele": modele.state_dict(),
            "optimiseur": optimiseur.state_dict(),
            "classes": CLASSES,
        }, temporaire)
        temporaire.replace(args.sortie)
        precision, rappel = evaluer(modele, val, appareil)
        print(f"Époque {epoque} : perte={perte_totale / len(train):.3f}, "
              f"précision={precision:.3f}, rappel={rappel:.3f} "
              f"(seuil de confiance 0,5 et IoU 0,5)", flush=True)
        print(f"Point de reprise sauvegardé : {args.sortie}", flush=True)


if __name__ == "__main__":
    main()
