# Acculturation IA

## Objectif

Cette fiche synthétise le support `01_acculturation_IA.pdf`.

Elle sert à situer les notions avant de manipuler les modèles dans InduSense : IA, Machine Learning, Deep Learning et IA générative.

## 1. Intelligence artificielle

L'intelligence artificielle désigne des systèmes capables de reproduire certaines tâches associées à l'intelligence humaine :

- raisonner à partir d'informations ;
- planifier une action ;
- reconnaître des formes ;
- produire une prédiction ;
- générer du texte, du code ou des images.

Dans un projet industriel, l'IA n'est pas magique : elle transforme des données et des règles d'apprentissage en aide à la décision.

## 2. Machine Learning

Le Machine Learning est une branche de l'IA.

Un modèle apprend à partir d'exemples. Il repère des relations dans les données, puis les utilise sur de nouvelles observations.

Dans InduSense :

```text
historique capteurs + incidents -> apprentissage -> score de panne future
```

Le modèle ne comprend pas la machine comme un technicien. Il apprend des régularités statistiques dans les données disponibles.

## 3. Deep Learning

Le Deep Learning est un sous-ensemble du Machine Learning basé sur des réseaux de neurones profonds.

Il est souvent pertinent pour :

- images ;
- audio ;
- texte ;
- grands volumes de données non structurées.

Pour InduSense, les données sont surtout tabulaires : capteurs, incidents, maintenance, machines. Des modèles comme régression logistique, arbre de décision, Random Forest ou XGBoost sont donc de bons premiers choix.

## 4. IA générative

L'IA générative produit du contenu à partir d'une demande :

- texte ;
- résumé ;
- code ;
- image ;
- assistant conversationnel.

Elle peut aider à documenter, expliquer ou accélérer l'analyse. Elle ne remplace pas l'évaluation métier du modèle.

## 5. Différence entre IA prédictive et IA générative

| usage | question | exemple InduSense |
|---|---|---|
| IA prédictive | que va-t-il probablement se passer ? | probabilité de panne dans 24 h |
| IA générative | que peut-on produire à partir d'un contexte ? | résumé d'un rapport de run ML |

## 6. À retenir

- L'IA est le domaine général.
- Le Machine Learning apprend à partir de données.
- Le Deep Learning est utile surtout sur des données complexes et massives.
- L'IA générative produit du contenu, mais ne valide pas un modèle.
- Dans InduSense, le coeur du POC est une IA prédictive supervisée.
