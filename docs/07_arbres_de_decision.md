# Arbres de décision

## Objectif

Cette fiche résume le support `04_Arbres_de_decisions_AELION.pdf` : principe des arbres, critères d'impureté, construction, généralisation, régression et limites.

Un arbre de décision découpe les données en posant une suite de questions simples.

## 1. Principe

Un arbre fonctionne par règles successives.

Exemple :

```text
temperature > 85 ?
  oui -> vibration > 0.7 ?
  non -> classe normale
```

Chaque noeud contient une question.

Chaque branche correspond à une réponse.

Chaque feuille donne une prédiction.

## 2. Classification

En classification, l'arbre cherche à séparer les classes.

Exemple :

| température | vibration | classe |
|---:|---:|---|
| 55 | 0.2 | normal |
| 91 | 0.8 | panne |

Un bon split crée des groupes plus homogènes.

## 3. Mesurer l'impureté

Un noeud est pur si toutes ses lignes appartiennent à la même classe.

Critères fréquents :

| critère | idée |
|---|---|
| Gini | mesure le mélange des classes |
| Entropie | mesure l'incertitude |
| MSE | utilisé pour les arbres de régression |

L'arbre choisit le split qui réduit le plus l'impureté.

## 4. Construction de l'arbre

La construction est récursive.

Étapes :

1. Tester plusieurs variables et seuils.
2. Choisir le meilleur split.
3. Répéter sur chaque sous-groupe.
4. Arrêter selon des critères définis.

Critères d'arrêt possibles :

- profondeur maximale ;
- nombre minimal de lignes par feuille ;
- gain trop faible ;
- noeud déjà pur.

## 5. Partitionnement de l'espace

Un arbre découpe l'espace des variables en zones rectangulaires.

Chaque zone correspond à une feuille.

C'est très lisible, mais parfois rigide : une petite variation des données peut changer fortement l'arbre.

## 6. Généralisation

Un arbre trop profond apprend trop bien les données d'entraînement.

Il risque alors de mal généraliser.

Signes de surapprentissage :

- très bon score train ;
- mauvais score test ;
- nombreuses feuilles avec très peu d'exemples.

## 7. Régression

Un arbre peut aussi prédire une valeur numérique.

Dans ce cas, une feuille prédit souvent la moyenne des valeurs d'entraînement qui arrivent dans cette feuille.

Exemple :

```text
feuille -> durée moyenne avant panne = 18.4 heures
```

## Avantages et limites

Avantages :

- facile à expliquer ;
- accepte des relations non linéaires ;
- demande peu de préparation des variables ;
- utile pour explorer des règles métier.

Limites :

- instable ;
- sensible au surapprentissage ;
- moins performant seul qu'un ensemble d'arbres ;
- peut créer des découpages trop brutaux.

## À retenir

Un arbre de décision est un modèle simple et interprétable. Il est très utile pour comprendre des règles, mais il doit être contrôlé pour éviter le surapprentissage.
