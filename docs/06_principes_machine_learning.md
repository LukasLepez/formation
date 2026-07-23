# Principes du Machine Learning

## Objectif

Cette fiche synthétise le support `02_machine_learning.pdf`.

Elle introduit la logique d'apprentissage et d'exploitation utilisée ensuite dans les modèles InduSense.

## 1. Apprentissage et exploitation

Un projet Machine Learning se déroule en deux grands temps.

Pendant l'apprentissage, le modèle observe des exemples connus :

```text
features -> label connu
```

Pendant l'exploitation, le modèle reçoit de nouvelles features et produit une prédiction :

```text
features nouvelles -> prédiction
```

Dans InduSense, les features sont construites dans le Gold Dataset. Le label indique si une panne survient dans une fenêtre future.

## 2. Features et cible

Les features sont les variables utilisées pour prédire.

Exemples :

- température moyenne sur 1 h ;
- pression maximale ;
- incidents récents ;
- jours depuis la dernière maintenance ;
- z-score par machine.

La cible est ce que l'on veut prédire.

Exemples :

- `label_failure_next_6h` ;
- `label_failure_next_12h` ;
- `label_failure_next_24h` ;
- `label_failure_next_48h`.

## 3. Apprentissage supervisé

InduSense est un cas d'apprentissage supervisé : on possède des exemples historiques avec la réponse attendue.

Le modèle apprend sur le passé pour prédire le futur.

Point important : le split doit respecter le temps. On évite de mélanger passé et futur, sinon le score d'évaluation devient trop optimiste.

## 4. Classification binaire

La maintenance prédictive du POC est une classification binaire.

```text
0 -> pas de panne dans la fenêtre future
1 -> panne dans la fenêtre future
```

Le modèle produit souvent une probabilité. Ensuite, un seuil transforme cette probabilité en alerte.

```text
probabilité >= seuil -> alerte
probabilité < seuil  -> pas d'alerte
```

## 5. Cycle minimal dans InduSense

1. Construire un Gold Dataset propre.
2. Séparer train, validation et test dans l'ordre temporel.
3. Entraîner plusieurs baselines.
4. Choisir un seuil selon l'objectif métier.
5. Comparer les modèles sur recall, precision, F1, PR-AUC et matrice de confusion.
6. Documenter l'arbitrage.

## À retenir

- Le modèle apprend sur des exemples historiques.
- Les features décrivent la situation.
- Le label décrit l'événement à prédire.
- En maintenance, une panne manquée coûte souvent plus cher qu'une fausse alerte.
- Le choix du seuil fait partie du modèle opérationnel.
