# Test annoté par le modèle

Les résultats sont séparés en deux archives téléchargeables :

- [Scènes complètes](https://github.com/AlexandreAlkozai/Projet-IA-Alexandre-Alkozai/releases/download/test-annote-par-le-modele/Scenes_completes_resultats.zip) : 2 511 sorties originales du détecteur. Les boîtes sont des prédictions, pas des annotations de référence. Parmi ces sorties, 138 ne présentent aucune annotation visible.
- [Crops](https://github.com/AlexandreAlkozai/Projet-IA-Alexandre-Alkozai/releases/download/test-annote-par-le-modele/Crops_validation_reconstituee.zip) : 1 574 images avec la classe prédite et la confiance, également détaillées dans un fichier CSV.

Les crops proviennent d'une partition 80/20 recalculée avec la graine 42 sur les
fichiers disponibles. L'identité exacte avec la partition historique ne peut
pas être garantie. **Ce n'est pas un test indépendant.**
