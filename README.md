# Analyse de Sentiments avec BERT

## Description
Ce dépôt contient un script Python pour l'analyse de sentiments en utilisant BERT (Bidirectional Encoder Representations from Transformers) de Hugging Face. L'objectif de l'analyse de sentiments est de déterminer le sentiment exprimé dans un texte donné, qu'il soit positif ou négatif. BERT est un modèle de langage puissant qui peut fournir des informations précieuses sur le sentiment des données textuelles.

Dans ce projet, nous utilisons le dataset "IMDB Dataset of 50K Movie Reviews", qui contient des avis de films étiquetés comme positifs ou négatifs. Vous pouvez télécharger le dataset en cliquant [ici](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?datasetId=134715&sortBy=voteCount)

Nous prétraitons les données, les divisons en ensembles d'entraînement et de test, puis nous entraînons le modèle BERT pour prédire le sentiment des avis de films.

## Dépendances
Avant d'exécuter le script, assurez-vous d'installer les dépendances requises. Vous pouvez les installer avec `pip` en utilisant la commande suivante:


## Fichiers
1. **sentiment_analysis.py**: Ce script est le point d'entrée principal pour la tâche d'analyse de sentiments. Il effectue les étapes suivantes :
   - Chargement du jeu de données IMDB et échantillonnage d'un nombre spécifié d'avis pour l'analyse.
   - Nettoyage des avis en supprimant les balises HTML et les caractères spéciaux, puis les convertit en minuscules.
   - Préparation des données pour l'entraînement en convertissant le texte en tokens compatibles avec BERT et en encodant les étiquettes.
   - Chargement du tokenizer BERT et du modèle pré-entraîné BERT pour la classification de séquences.
   - Configuration de l'appareil (GPU si disponible, sinon CPU) pour l'entraînement du modèle.
   - Entraînement du modèle BERT en utilisant un traitement par lots (batch processing) et un optimiseur avec un programme de taux d'apprentissage.
   - Évaluation de l'exactitude du modèle et génération d'une matrice de confusion pour évaluer ses performances.

2. **data_processing.py**: Ce module contient des fonctions utilitaires pour le chargement et le prétraitement des données. Il comprend les fonctions suivantes :
   - `load_dataset(file_path)`: Charge le jeu de données IMDB à partir du fichier CSV donné.
   - `clean_review(review)`: Nettoie un avis en supprimant les balises HTML, les caractères spéciaux et en le convertissant en minuscules.
   - `prep_train_test(data, test_size=0.2, random_state=42)`: Divise les données en ensembles d'entraînement et de test avec la taille de test et l'état aléatoire spécifiés.

3. **batch_processing.py**: Ce module contient la fonction `train_batch()`, qui effectue l'entraînement par lots pour le modèle BERT. Il s'occupe de la tokenisation, de l'allocation de l'appareil et des mises à jour des gradients.

## Utilisation
Pour utiliser le script d'analyse de sentiments, suivez ces étapes :
1. Téléchargez le fichier IMDB Dataset.csv et placez-le dans le même répertoire que le script sentiment_analysis.py.
2. Assurez-vous d'avoir Python 3.x et PyTorch installés.
3. Installez les dépendances requises comme mentionné ci-dessus.
4. Exécutez le script sentiment_analysis.py en utilisant la commande suivante :


## Configuration
Vous pouvez ajuster les paramètres suivants dans sentiment_analysis.py pour répondre à vos besoins spécifiques :
- `num_reviews` : Le nombre d'avis à échantillonner à partir du jeu de données pour l'analyse.
- `batch_size` : La taille de chaque lot d'entraînement pour le modèle BERT.
- `num_epochs` : Le nombre d'epoch d'entraînement pour le modèle BERT. (Une epoch correspond à un passage du modèle sur l'ensemble de l'échantillon sélectionné)
- `learning_rate` : Le taux d'apprentissage utilisé par l'optimiseur pendant l'entraînement.
- `test_size` : Le pourcentage de données à utiliser pour les tests lors du processus de division des données.

## Note importante
Veuillez ajuster la taille du lot (batch size), le nombre d'epochs et le taux d'apprentissage en fonction de votre jeu de données spécifique et des ressources informatiques disponibles. De plus, envisagez de modifier le processus de nettoyage ou d'utiliser un autre tokenizer en fonction de la nature de votre jeu de données.

## Résultats et Évaluation
Le script fournit les sorties suivantes :
- Progrès de l'entraînement : Le script affiche la perte d'entraînement pour chaque epoch pendant le processus d'entraînement.
- Évaluation du modèle : Après l'entraînement, le script affiche l'exactitude du modèle sur l'ensemble de test et affiche une matrice de confusion pour évaluer ses performances.
