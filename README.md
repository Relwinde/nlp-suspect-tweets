# nlp-suspect-tweets
Exercice de NLP Sur la détection de tweet malveillant

# Détection de tweets suspects

## Description

Ce notebook explore différentes techniques de Machine Learning pour la détection de tweets suspects. Il utilise un jeu de données de tweets étiquetés comme suspects ou non suspects et applique des méthodes de traitement du langage naturel (NLP) pour extraire des caractéristiques pertinentes. 

Plusieurs modèles de classification sont entraînés et comparés, notamment la régression logistique, les machines à vecteurs de support (SVM), les forêts aléatoires (Random Forest), les arbres de décision, les K-plus proches voisins (KNN) et XGBoost. Le modèle le plus performant est ensuite évalué à l'aide de métriques telles que le F1-score, la matrice de confusion et la courbe ROC-AUC.

## Fonctionnalités

* **Exploration du jeu de données:** Analyse descriptive des tweets, visualisation de la distribution des classes et exploration de la polarité des discours.
* **Prétraitement du texte:** Nettoyage, suppression des mots vides (stop words), lemmatisation.
* **Extraction de caractéristiques:** Utilisation de DistilBERT pour générer des représentations vectorielles des tweets (embeddings).
* **Gestion des classes déséquilibrées:** Application de la technique SMOTE pour sur-échantillonner la classe minoritaire.
* **Entraînement et évaluation des modèles:** Utilisation de plusieurs algorithmes de classification et comparaison de leurs performances.
* **Sélection du meilleur modèle:** Choix du modèle le plus performant en fonction des métriques d'évaluation.
* **Sauvegarde du modèle:** Enregistrement du modèle sélectionné pour une utilisation ultérieure.

## Dépendances

* Python 3.x
* Pandas
* Scikit-learn
* NLTK
* TextBlob
* Transformers
* Imbalanced-learn
* XGBoost
* Seaborn
* Matplotlib
* Joblib

## Installation

1. Cloner le référentiel : `git clone <https://github.com/Relwinde/nlp-suspect-tweets>`

## Utilisation

1. Ouvrir le notebook dans Google Colab.
3. Charger le fichier tweets_suspects.csv dans le stockage de la session
2. Exécuter les cellules du notebook dans l'ordre.
3. Le modèle entraîné est sauvegardé dans le fichier `rf_model.pkl`.

## Auteurs

* [Your Name]

## Licence

[Specify the license, e.g., MIT License]
