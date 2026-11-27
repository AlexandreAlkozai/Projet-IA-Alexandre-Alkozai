# AeroScope : classification et détection d'avions

L'application Streamlit propose deux modes : **Crop** classe une image contenant
un avion, et **Scène complète** détecte et classe les avions présents dans une
image. Les résultats dépendent des modèles fournis et de leurs données
d'entraînement ; ce projet ne fournit pas de mesure de performance garantie.

## Où trouver les programmes ?

1. **Analyse du jeu de données** : [programme d'analyse](01_analyse_du_dataset/analyse_dataset.py).
2. **Création des images recadrées** : [programme de création des crops](02_creation_des_crops/creation_crops.py).
3. **Classification des avions** : [entraînement du classifieur](03_classification/entrainement_classifieur.py).
4. **Détection dans une scène complète** : [entraînement du détecteur](04_detection_des_scenes/entrainement_detection.py), qui utilise [detection_avions.py](fichiers_techniques/detection_avions.py).

**Résultats annotés** : voir [Test annoté par le modèle](Test%20annot%C3%A9%20par%20le%20mod%C3%A8le), avec une archive pour les scènes complètes et une pour les crops.

**Site / application Streamlit** : le code est dans
[app_detection.py](app_detection.py), à la racine du dépôt. Ce lien ouvre le
code, pas une application en ligne. Les
étapes de lancement local et de déploiement sont indiquées ci-dessous.

Le dossier [fichiers_techniques](fichiers_techniques) regroupe le code du
détecteur et les modèles `detecteur_8_epoques.pt` (environ 331 Mo) et
`modele_avions.pt` (environ 95 Mo). Le détecteur contient également l'état de
son optimiseur. Les fichiers de configuration nécessaires restent à la racine.
Ces fichiers sont regroupés, mais restent publics.

Les modèles sont des copies inchangées des fichiers fournis. Les quatre images
de la bannière sont dans `images_banniere`. Le jeu de données n'est pas inclus.

## Utilisation locale sous Windows

Avec Python 3.15 et Git LFS installés, ouvrir PowerShell dans ce dossier :

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app_detection.py
```

Ouvrir l'adresse locale affichée, choisir un mode et importer une image JPG ou
PNG. Si le projet a été cloné, récupérer les véritables modèles avec Git LFS :
`git lfs pull`. Sans eux, les fichiers `.pt` peuvent être de simples pointeurs.
Ne charger que des checkpoints de confiance.

## Modèles et Git LFS

Le détecteur dépasse la limite GitHub de 100 MiB pour un fichier Git ordinaire.
Le fichier `.gitattributes` prépare donc **les deux fichiers `.pt` pour Git LFS**.
Ne pas les ajouter par l'interface web de GitHub. Vérifier les droits de
publication du code et des modèles, le quota et le budget LFS avant une mise à jour.
Les téléchargements et déploiements consomment la bande passante du propriétaire
du dépôt ; les nouvelles versions des modèles consomment du stockage
supplémentaire.

Le téléchargement ZIP GitHub ne contient pas forcément les objets LFS. Pour
obtenir les modèles, préférer `git clone` avec Git LFS installé puis
`git lfs pull`. Le dossier `dataset` n'est pas publié par défaut.

## Déploiement Streamlit Community Cloud, après publication autorisée

1. Se connecter à [Streamlit Community Cloud](https://share.streamlit.io/) avec
   le compte autorisé à administrer le dépôt.
2. Sélectionner **Create app**, puis le dépôt public, la branche `main` et le
   fichier `app_detection.py`. Choisir éventuellement un sous-domaine.
3. Dans **Advanced settings**, sélectionner Python **3.15**, puis lancer le
   déploiement. La documentation Streamlit indique la prise en charge de Git
   LFS ; `requirements.txt` demande les versions CPU de PyTorch disponibles
   via son index.
4. Tester les deux modes avec des images dont le partage est autorisé, puis
   transmettre au professeur l'URL du dépôt et celle de l'application.

Le déploiement réel, sa disponibilité et les ressources mémoire/CPU de la
plateforme restent à vérifier après publication ; la détection peut être lente
sur CPU.

## Scripts du projet

Lancer les scripts individuellement avec Python et suivre leurs invites pour
indiquer le jeu de données local. Par exemple, depuis la racine du dépôt :

```powershell
.\.venv\Scripts\python.exe 01_analyse_du_dataset\analyse_dataset.py
```

Le détecteur `detection_avions.py` accepte aussi `--dataset` ; sans ce paramètre,
il cherche un dossier `dataset` à la racine. Le programme de l'étape 4 lance
un entraînement de huit époques. Celui de l'étape 3 lance un entraînement et
**écrase** le fichier `fichiers_techniques/modele_avions.pt` à la fin :
conserver une sauvegarde avant de l'utiliser.
Les entraînements peuvent télécharger des poids préentraînés et nécessitent le
jeu de données, qui n'est pas fourni ici.

## Références

- [Limites de fichiers GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Facturation et quotas Git LFS](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [Organisation des fichiers et prise en charge de Git LFS sur Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Dépendances Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Déploiement Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
