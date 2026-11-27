# AeroScope : classification et détection d'avions

L'application Streamlit propose deux modes : **Crop** classe une image contenant
un avion, et **Scène complète** détecte et classe les avions présents dans une
image. Les résultats dépendent des modèles fournis et de leurs données
d'entraînement ; ce projet ne fournit pas de mesure de performance garantie.

## Contenu

- `app_detection.py` : interface Streamlit.
- `detection_avions.py` : architecture et entraînement du détecteur.
- `detecteur_8_epoques.pt` : détecteur fourni, avec état d'optimiseur (environ
  331 Mo).
- `modele_avions.pt` : classifieur fourni (environ 95 Mo).
- `analyse_dataset.py`, `creation_crops.py`, `entrainement_classifieur.py` et
  `entrainement_detection.py` : les quatre scripts d'étapes fournis.
- `requirements.txt` : dépendances Python de l'application et des scripts.

Les modèles sont des copies inchangées des fichiers fournis. Le jeu de données
et l'image de bannière ne sont pas inclus. L'application fonctionne sans
bannière ; pour en ajouter une, placer une image JPEG autorisée nommée
`f14_nasa.jpg` à côté de `app_detection.py`.

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

## Publication GitHub, seulement après autorisation

Le détecteur dépasse la limite GitHub de 100 MiB pour un fichier Git ordinaire.
Le fichier `.gitattributes` prépare donc **les deux fichiers `.pt` pour Git LFS**.
Ne pas les ajouter par l'interface web de GitHub. Vérifier les droits de
publication du code et des modèles, le quota et le budget LFS avant de publier.
Les téléchargements et déploiements consomment la bande passante du propriétaire
du dépôt ; les nouvelles versions des modèles consomment du stockage
supplémentaire.

1. Choisir le compte GitHub et le nom du dépôt ; après autorisation, créer un
   dépôt **public vide**, sans README ni autre fichier initial.
2. Dans PowerShell, depuis ce dossier, préparer le dépôt local et vérifier que
   les deux modèles apparaissent dans `git lfs ls-files` :

   ```powershell
   git init -b main
   git lfs install --local
   git add .
   git lfs ls-files
   git status --short
   git commit -m "Ajouter AeroScope"
   ```

3. Après confirmation explicite de l'envoi public, remplacer l'URL ci-dessous par
   celle du dépôt vide créé sur GitHub :

   ```powershell
   git remote add origin https://github.com/COMPTE/NOM-DU-DEPOT.git
   git push -u origin main
   ```

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
indiquer le jeu de données local. Le détecteur accepte aussi `--dataset` ; sans
ce paramètre, il cherche un dossier `dataset` à côté du script. Le script
`entrainement_detection.py` lance un entraînement de huit époques.
`entrainement_classifieur.py` lance un entraînement et **écrase**
`modele_avions.pt` à la fin : conserver une sauvegarde avant de l'utiliser.
Les entraînements peuvent télécharger des poids préentraînés et nécessitent le
jeu de données, qui n'est pas fourni ici.

## Références

- [Limites de fichiers GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Facturation et quotas Git LFS](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [Organisation des fichiers et prise en charge de Git LFS sur Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Dépendances Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Déploiement Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
