from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# Demander le dossier qui contient les sous-dossiers A1, A2, etc.
dossier_crops = Path(
    input("Chemin du dossier crops_par_classe : ").strip().strip('"')
)

if not dossier_crops.is_dir():
    raise FileNotFoundError(f"Dossier introuvable : {dossier_crops}")

# Préparer les images comme attendu par ResNet50
preparation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ImageFolder utilise le nom des sous-dossiers comme étiquette
dataset = datasets.ImageFolder(dossier_crops, transform=preparation)

if len(dataset.classes) < 2:
    raise ValueError("Il faut au moins deux dossiers de classes dans crops_par_classe.")

print(f"{len(dataset)} images chargées")
print(f"Classes : {dataset.classes}")

# Séparer les données : 80 % entraînement, 20 % validation
taille_entrainement = int(0.8 * len(dataset))
taille_validation = len(dataset) - taille_entrainement

entrainement, validation = random_split(
    dataset,
    [taille_entrainement, taille_validation],
    generator=torch.Generator().manual_seed(42),
)

chargeur_entrainement = DataLoader(
    entrainement, batch_size=16, shuffle=True, num_workers=0
)
chargeur_validation = DataLoader(
    validation, batch_size=16, shuffle=False, num_workers=0
)

# Utiliser le GPU s'il est disponible, sinon le CPU
appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {appareil}")

# Charger ResNet50 avec ses poids pré-entraînés, puis adapter la dernière couche
modele = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
modele.fc = nn.Linear(modele.fc.in_features, len(dataset.classes))
modele = modele.to(appareil)

fonction_perte = nn.CrossEntropyLoss()
optimiseur = torch.optim.Adam(modele.parameters(), lr=0.0001)

# Entraîner et mesurer la précision de validation
nombre_epoques = 8

for epoque in range(nombre_epoques):
    modele.train()
    perte_totale = 0.0

    for images, etiquettes in chargeur_entrainement:
        images = images.to(appareil)
        etiquettes = etiquettes.to(appareil)

        optimiseur.zero_grad()
        predictions = modele(images)
        perte = fonction_perte(predictions, etiquettes)
        perte.backward()
        optimiseur.step()

        perte_totale += perte.item()

    modele.eval()
    bonnes_predictions = 0
    total = 0

    with torch.no_grad():
        for images, etiquettes in chargeur_validation:
            images = images.to(appareil)
            etiquettes = etiquettes.to(appareil)

            predictions = modele(images)
            classes_predites = predictions.argmax(dim=1)

            bonnes_predictions += (classes_predites == etiquettes).sum().item()
            total += etiquettes.size(0)

    precision = 100 * bonnes_predictions / total
    perte_moyenne = perte_totale / len(chargeur_entrainement)

    print(
        f"Époque {epoque + 1}/{nombre_epoques} - "
        f"perte : {perte_moyenne:.4f} - "
        f"précision validation : {precision:.2f} %"
    )

# Enregistrer le modèle dans le format attendu par l'application
chemin_modele = Path(__file__).resolve().parent / "modele_avions.pt"

torch.save(
    {
        "poids": modele.cpu().state_dict(),
        "classes": dataset.classes,
    },
    chemin_modele,
)

print(f"Modèle enregistré ici : {chemin_modele}")