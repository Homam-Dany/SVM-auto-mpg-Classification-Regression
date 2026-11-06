# Prédiction de la consommation de carburant avec les SVM (Auto MPG)

Dans ce projet, j'essaie de prédire la consommation de carburant de voitures à partir de leurs caractéristiques techniques (poids, puissance, cylindrée, année...). Je traite le même problème de deux façons : comme une **classification** (SVC) puis comme une **régression** (SVR). Dans les deux cas, j'implémente d'abord le modèle moi-même avec numpy, puis je compare mes résultats avec ceux de scikit-learn.

---

## 1. Sujet du devoir

L'énoncé demande de :

- approximer la consommation (`mpg`, miles per gallon) du jeu de données **Auto MPG** avec des SVM ;
- pour **chaque approche**, proposer un prétraitement justifié, une implémentation depuis zéro et une comparaison avec scikit-learn ;
- en **classification** : transformer `mpg` en classes (faible, moyenne, élevée), discuter l'impact du choix des seuils sur l'équilibre des classes et la difficulté du problème, puis implémenter un SVC linéaire et un SVC à noyau non linéaire ;
- en **régression** : utiliser un SVR linéaire avec la perte ε-insensible et étudier l'influence de ε sur le nombre de vecteurs de support et sur la capacité du modèle à ignorer le bruit.

> Remarque : plus `mpg` est élevé, moins la voiture consomme. Une classe « mpg élevé » correspond donc à une consommation **faible**.

---

## 2. Les données

Le jeu de données vient de seaborn :

```python
import seaborn as sns
df = sns.load_dataset("mpg")
```

Il contient des voitures des années 70-80, avec :

| Type | Variables |
|------|-----------|
| Cible | `mpg` |
| Continues | `displacement`, `horsepower`, `weight`, `acceleration`, `model_year` |
| Catégorielles | `cylinders`, `origin` (usa, japan, europe) |
| Non utilisée | `name` (nom de la voiture) |

La colonne `horsepower` a 6 valeurs manquantes. Je supprime ces lignes, ce qui donne **392 voitures** comme dans l'énoncé.

---

## 3. Contenu du notebook

| Partie | Contenu |
|--------|---------|
| 1 | Introduction, rappel sur les SVM, imports, chargement et nettoyage |
| 2 | Exploration des données : distributions, corrélations, nuages de points, boxplots |
| 3 | Prétraitement : encodage one-hot, découpage train/test, standardisation |
| 4 | Discrétisation de `mpg` en 3 classes et comparaison de trois stratégies |
| 5 | SVC linéaire from scratch (perte hinge, descente de sous-gradient, One-vs-Rest) |
| 6 | SVC à noyau RBF from scratch, étude de l'influence de γ |
| 7 | SVR linéaire from scratch avec perte ε-insensible |
| 8 | Influence de ε : vecteurs de support et robustesse au bruit |
| 9 | Récapitulatif, conclusions et limites |
| 10 | Sources et répartition du travail |

---

## 4. Démarche

### Prétraitement

- **One-hot encoding** de `origin`, pour ne pas imposer un ordre artificiel entre les pays. `cylinders` est gardée comme variable numérique, car ses valeurs sont ordonnées.
- **Découpage 80 % / 20 %** (train / test) avec `random_state=42`.
- **Standardisation** de chaque variable numérique. La moyenne et l'écart-type sont calculés **uniquement sur le train** puis appliqués au test, pour éviter la fuite de données (*data leakage*). J'ai vérifié que mon calcul à la main donne les mêmes valeurs que `StandardScaler`.

### Classification (SVC)

Pour transformer `mpg` en 3 classes, j'ai comparé trois stratégies de discrétisation :

1. intervalles de **même largeur** ;
2. **quantiles** (même nombre de voitures par classe) ;
3. **seuils fixés à la main** (20 et 30 mpg).

J'ai retenu les **quantiles**, calculés sur le train, car ils donnent des classes équilibrées et s'adaptent à la forme de la distribution. Le tableau comparatif et la discussion sont dans la partie 4 du notebook.

Un SVM étant binaire par nature, j'utilise la stratégie **One-vs-Rest** : un SVM par classe, et on prédit la classe dont le score est le plus élevé.

### Régression (SVR)

Le SVR cherche une droite autour de laquelle on trace un « tube » de largeur 2ε. Les erreurs à l'intérieur du tube ne sont pas pénalisées. La cible `mpg` est standardisée (avec les statistiques du train) pour que la descente de gradient soit stable, puis les prédictions sont ramenées en mpg pour calculer les métriques.

---

## 5. Implémentation from scratch

Tous les modèles sont codés avec numpy, par descente de sous-gradient sur la forme primale.

**SVC linéaire** : on minimise

```
J(w, b) = (λ/2)·||w||² + (1/n)·Σ max(0, 1 − yᵢ(w·xᵢ + b))
```

**SVC à noyau RBF** : le noyau est `K(x, x') = exp(−γ·||x − x'||²)`. La fonction de décision s'écrit `f(x) = Σ αⱼ K(xⱼ, x) + b` et on apprend les coefficients α.

**SVR linéaire** : on minimise

```
J(w, b) = (λ/2)·||w||² + (1/n)·Σ max(0, |yᵢ − w·xᵢ − b| − ε)
```

Pour comparer avec scikit-learn, j'utilise la correspondance **C = 1 / (λ·n)**.

Les résultats ne sont pas strictement identiques à ceux de scikit-learn. C'est normal : scikit-learn résout le problème dual de façon exacte (LIBSVM), alors que ma version est une descente de gradient avec un nombre fixé d'itérations.

---

## 6. Résultats

> *(À compléter avec mes propres résultats après exécution du notebook.)*

**Classification** (3 classes, seuils par quantiles : ... et ... mpg)

| Modèle | Accuracy train | Accuracy test |
|--------|----------------|---------------|
| SVC linéaire (from scratch) | ... | ... |
| SVC linéaire (scikit-learn) | ... | ... |
| SVC RBF (from scratch) | ... | ... |
| SVC RBF (scikit-learn) | ... | ... |

**Régression** (ε = 0,1)

| Modèle | MAE test (mpg) | RMSE test (mpg) | R² test |
|--------|----------------|-----------------|---------|
| SVR linéaire (from scratch) | ... | ... | ... |
| SVR linéaire (scikit-learn) | ... | ... | ... |
| Moindres carrés (référence) | ... | ... | ... |

**Ce que j'ai observé** *(à rédiger avec mes mots)* :

- l'effet du noyau RBF par rapport au modèle linéaire : ...
- l'effet de γ (sous-apprentissage / surapprentissage) : ...
- l'évolution du nombre de vecteurs de support quand ε augmente : ...
- le comportement du SVR face au bruit et aux valeurs aberrantes : ...

---

## 7. Exécuter le projet

### Prérequis

- Python 3.9 ou plus
- Les bibliothèques : `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

### Installation

```bash
git clone https://github.com/TON_PSEUDO/svm-auto-mpg-classification-regression.git
cd svm-auto-mpg-classification-regression
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook SVM_Auto_MPG_SVC_SVR.ipynb
```

Ensuite, dans Jupyter : **Kernel → Restart & Run All**.

### Sans rien installer

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TON_PSEUDO/svm-auto-mpg-classification-regression/blob/main/SVM_Auto_MPG_SVC_SVR.ipynb)

Le dataset est téléchargé automatiquement par seaborn, il faut donc une connexion internet à la première exécution.

---

## 8. Structure du dépôt

```
svm-auto-mpg-classification-regression/
├── SVM_Auto_MPG_SVC_SVR.ipynb   # notebook complet (explications + code)
└── README.md                    # ce fichier
```

---

## 9. Limites et pistes d'amélioration

- Les hyperparamètres (λ, γ, ε) sont fixés à la main. Une validation croisée permettrait de les choisir plus rigoureusement.
- Le SVR est linéaire. Un SVR à noyau RBF pourrait mieux capter la relation courbe entre `mpg` et `weight`.
- Un seul découpage train/test est utilisé, donc les résultats dépendent un peu du tirage.
- Le comptage des vecteurs de support dans ma version est approché (la descente de gradient n'atterrit jamais exactement sur le bord du tube), alors que celui de scikit-learn est exact.

---

## 10. Sources

- Cours de Machine Learning (SVM, noyaux, SVR) et travaux dirigés associés.
- Documentation de scikit-learn : https://scikit-learn.org/stable/modules/svm.html
- Jeu de données Auto MPG, chargé via `seaborn.load_dataset('mpg')`.
- Bibliothèques : numpy, pandas, matplotlib, seaborn, scikit-learn.

## 11. Répartition du travail

**Travail réalisé de manière autonome.** Chaîne de traitement des données (chargement, nettoyage, exploration, encodage, standardisation, découpage), stratégie de discrétisation de la cible, métriques d'évaluation, comparaison avec scikit-learn, interprétation des résultats et conclusions.

**Travail réalisé avec l'appui d'un assistant IA (Claude).** Formulation mathématique et implémentation du SVC à noyau (coefficients α, matrice de Gram), sous-gradient du SVR avec perte ε-insensible, conception du protocole d'étude de la robustesse au bruit. Ce code a été relu, testé, adapté et validé par comparaison avec scikit-learn. L'assistant a aussi servi à clarifier certaines notions et à améliorer la formulation de certains passages.

---

*Projet réalisé dans le cadre du module de Machine Learning, FST Tanger.*
