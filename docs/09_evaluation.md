# Évaluation des modèles

## Objectif

Cette fiche résume le support `06_Evaluation_AELION.pdf` : test set, ordre caché, métriques de régression, matrice de confusion, précision, rappel et ROC AUC.

Évaluer un modèle sert à mesurer sa capacité à généraliser sur des données qu'il n'a pas vues pendant l'entraînement.

## 1. Test set

Le test set est une partie des données gardée à part.

Il ne doit pas servir à entraîner le modèle.

Découpage classique :

```text
train -> apprendre
validation -> choisir les réglages
test -> estimer la performance finale
```

Le test set doit représenter les conditions réelles d'utilisation.

## 2. Attention aux ordres cachés

Certaines données ont un ordre naturel.

Exemples :

- séries temporelles ;
- relevés machines ;
- historiques clients ;
- lots de production.

Dans ce cas, un split aléatoire peut créer une fuite d'information.

Pour des données temporelles, il vaut mieux entraîner sur le passé et tester sur le futur.

## 3. Évaluer une régression

Pour une régression, on compare des valeurs prédites à des valeurs réelles.

Métriques fréquentes :

| métrique | interprétation |
|---|---|
| MAE | erreur moyenne absolue |
| MSE | erreur quadratique moyenne |
| RMSE | racine de la MSE, dans l'unité de la cible |
| R² | part de variance expliquée |

Le RMSE pénalise fortement les grosses erreurs.

Le MAE est souvent plus facile à expliquer métier.

## 4. R² score

Le R² mesure si le modèle fait mieux qu'une prédiction moyenne.

Repères :

```text
R² proche de 1 -> très bon ajustement
R² proche de 0 -> pas mieux que la moyenne
R² négatif     -> pire que la moyenne
```

Un bon R² ne suffit pas toujours : il faut aussi regarder les erreurs et les cas extrêmes.

## 5. Évaluer une classification

Pour une classification, on compare des classes prédites à des classes réelles.

La matrice de confusion distingue :

| cas | sens |
|---|---|
| TP | vrai positif |
| TN | vrai négatif |
| FP | faux positif |
| FN | faux négatif |

Dans un contexte de panne, un faux négatif peut être plus grave qu'un faux positif.

## 6. Précision et rappel

La précision répond à la question :

```text
Parmi les alertes déclenchées, combien étaient correctes ?
```

Le rappel répond à la question :

```text
Parmi les vraies pannes, combien ont été détectées ?
```

Sur une classe rare, l'accuracy peut être trompeuse. Il faut regarder précision, rappel et F1-score.

## 7. ROC AUC

La courbe ROC compare :

- le taux de vrais positifs ;
- le taux de faux positifs.

L'AUC résume cette courbe en un score.

Plus l'AUC est proche de `1`, meilleure est la séparation entre les classes.

Pour des classes très déséquilibrées, une courbe Precision-Recall peut être plus informative que la ROC.

## À retenir

- Toujours garder un vrai jeu de test.
- Respecter l'ordre temporel quand il existe.
- Choisir les métriques selon le coût métier des erreurs.
- Ne pas se contenter de l'accuracy sur des classes déséquilibrées.
- Lire la matrice de confusion avant de conclure.
