# Prédiction de la consommation de carburant avec les SVM

Ce projet étudie comment les caractéristiques d'une voiture peuvent aider à estimer sa consommation. Le travail est présenté dans le notebook [Prédiction de la consommation de carburant avec les SVM.ipynb](./Prédiction%20de%20la%20consommation%20de%20carburant%20avec%20les%20SVM.ipynb).

## Données et objectif

Le notebook utilise le jeu de données **Auto MPG**, disponible avec Seaborn. Il contient notamment le poids, la puissance, la cylindrée et l'année des voitures. La consommation est exprimée en **mpg** (*miles per gallon*) : une valeur élevée correspond à une consommation plus faible.

Deux problèmes sont étudiés :

- **Classification avec SVC** : répartir les voitures en trois catégories de consommation.
- **Régression avec SVR** : prédire directement la valeur de consommation en mpg.

## Travail réalisé

Le notebook suit les principales étapes d'un projet de machine learning :

1. Découverte, exploration et nettoyage des données.
2. Préparation des variables, encodage des catégories et standardisation.
3. Séparation des données en ensembles d'entraînement et de test.
4. Implémentation de modèles SVM en NumPy, puis comparaison avec scikit-learn.
5. Étude des SVC linéaires et à noyau RBF, et du SVR linéaire.
6. Évaluation des modèles et exploration de l'influence du paramètre ε et du bruit.

## Résultats principaux

Sur l'ensemble de test, la meilleure précision de classification de ce tableau est obtenue par le **SVC RBF codé à la main** (0,810). Le SVC RBF de scikit-learn obtient 0,797.

Pour la régression, le SVR de scikit-learn obtient une MAE de **2,366 mpg**, une RMSE de **3,347 mpg** et un R² de **0,780**. Les résultats du SVR codé à la main sont proches (MAE 2,384 mpg, RMSE 3,371 mpg, R² 0,777). Le notebook compare aussi ces résultats à une régression par moindres carrés.

Ces chiffres sont ceux des résultats affichés dans le notebook et dépendent du découpage des données et des paramètres choisis.

## Exécution

Ouvrir le notebook avec Jupyter Notebook, JupyterLab ou VS Code, puis exécuter les cellules dans l'ordre. Les bibliothèques utilisées sont :

- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn

Le notebook charge le jeu de données avec `seaborn.load_dataset("mpg")`. Selon l'installation, une connexion Internet peut être nécessaire au premier chargement.

## Sources

- Jeu de données Auto MPG, chargé avec Seaborn.
- [Documentation scikit-learn sur les SVM](https://scikit-learn.org/stable/modules/svm.html)
- Cours et travaux dirigés du module d'apprentissage automatique.
