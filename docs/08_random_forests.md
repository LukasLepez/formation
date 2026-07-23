# Random Forests

## Objectif

Cette fiche résume le support `05_Random_Forests_AELION.pdf` : problème des arbres seuls, bagging, forêt aléatoire et avantages.

Une Random Forest combine plusieurs arbres de décision pour obtenir un modèle plus stable.

## 1. Limite des arbres de décision

Un arbre seul peut être très sensible aux données.

Si l'échantillon change un peu, l'arbre peut choisir d'autres splits et produire une structure différente.

Conséquence :

- forte variance ;
- risque de surapprentissage ;
- prédictions instables.

## 2. Bagging

Le bagging consiste à entraîner plusieurs modèles sur des échantillons différents du dataset.

Chaque échantillon est tiré avec remise.

Idée :

```text
dataset -> échantillon 1 -> arbre 1
dataset -> échantillon 2 -> arbre 2
dataset -> échantillon 3 -> arbre 3
```

Ensuite, on agrège les prédictions.

## 3. Agrégation

Pour une classification :

```text
classe finale = vote majoritaire des arbres
```

Pour une régression :

```text
valeur finale = moyenne des prédictions
```

Cette moyenne réduit les variations d'un arbre à l'autre.

## 4. Random Forest

Une Random Forest ajoute une deuxième source d'aléatoire.

À chaque split, l'arbre ne regarde qu'un sous-ensemble des variables.

Cela force les arbres à être différents les uns des autres.

Résultat :

- moins de variance ;
- meilleure généralisation ;
- modèle robuste sur beaucoup de problèmes tabulaires.

## 5. Paramètres importants

| paramètre | rôle |
|---|---|
| `n_estimators` | nombre d'arbres |
| `max_depth` | profondeur maximale |
| `min_samples_leaf` | taille minimale d'une feuille |
| `max_features` | nombre de variables testées à chaque split |
| `class_weight` | pondération des classes |

Pour une classe rare, comme une panne, `class_weight="balanced"` peut aider.

## 6. Avantages

- fonctionne bien sans beaucoup de réglages ;
- robuste aux relations non linéaires ;
- moins instable qu'un arbre seul ;
- donne une importance approximative des variables ;
- adapté aux données tabulaires.

## 7. Limites

- moins interprétable qu'un arbre seul ;
- peut être plus lourd à entraîner ;
- extrapole mal hors des valeurs observées ;
- les importances de variables peuvent être biaisées.

## À retenir

Une Random Forest est un ensemble d'arbres. Elle garde la flexibilité des arbres de décision, mais réduit leur instabilité grâce au bagging et à la sélection aléatoire des variables.
