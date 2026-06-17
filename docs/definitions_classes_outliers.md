# Définitions : classe et outlier

## Objectif

Cette note définit deux notions importantes en analyse de données et en machine learning :

- une classe ;
- un outlier.

Ces deux notions apparaissent souvent pendant la préparation d'un dataset, avant l'analyse ou l'entraînement d'un modèle.

## 1. Qu'est-ce qu'une classe ?

Une classe est une catégorie à laquelle appartient une ligne du dataset.

En machine learning supervisé, la classe correspond souvent à ce que l'on veut prédire.

Exemples :

| Contexte | Classes possibles |
|---|---|
| Détection de panne | `panne`, `pas de panne` |
| Gravité d'incident | `faible`, `moyenne`, `critique` |
| Qualité produit | `conforme`, `non conforme` |
| Type d'incident | `surchauffe`, `vibration`, `baisse pression` |

La colonne qui contient la classe est souvent appelée :

- variable cible ;
- target ;
- label ;
- classe à prédire.

Exemple :

| machine_id | temperature_c | pressure_bar | classe |
|---|---:|---:|---|
| MACH-01 | 52.1 | 198.4 | pas de panne |
| MACH-02 | 91.8 | 170.2 | panne |

Ici, la colonne `classe` indique la catégorie associée à chaque ligne.

Si l'objectif est de construire un modèle, cette colonne indique ce que le modèle doit apprendre à prédire.

## 2. Classification binaire

Une classification est binaire quand il existe seulement deux classes.

Exemples :

- panne / pas panne ;
- conforme / non conforme ;
- critique / non critique ;
- anomalie / normal.

Exemple :

```text
is_breakdown = True  -> panne
is_breakdown = False -> pas de panne
```

## 3. Classification multiclasse

Une classification est multiclasse quand il existe plus de deux classes.

Exemple :

```text
type_incident =
- surchauffe
- vibration
- bruit mécanique
- alarme capteur
- défaut qualité
```

Dans ce cas, le modèle doit choisir une classe parmi plusieurs catégories possibles.

## 4. Classe majoritaire et classe minoritaire

Dans un dataset, certaines classes peuvent être beaucoup plus fréquentes que d'autres.

La classe majoritaire est la classe la plus présente.

La classe minoritaire est la classe la moins présente.

Exemple :

| classe | nombre de lignes |
|---|---:|
| pas de panne | 9 500 |
| panne | 500 |

Ici :

- `pas de panne` est la classe majoritaire ;
- `panne` est la classe minoritaire.

Ce déséquilibre est important, car un modèle peut apprendre à prédire surtout la classe majoritaire et ignorer la classe rare.

Dans un contexte industriel, la classe rare est pourtant souvent la plus importante : panne, incident critique, défaut qualité grave ou anomalie capteur.

## 5. Qu'est-ce qu'un outlier ?

Un outlier est une valeur atypique.

C'est une donnée qui s'écarte fortement du comportement général des autres données.

On peut aussi parler de :

- valeur extrême ;
- valeur anormale ;
- valeur suspecte ;
- valeur aberrante.

Exemple :

| machine_id | temperature_c |
|---|---:|
| MACH-01 | 50.2 |
| MACH-02 | 51.1 |
| MACH-03 | 49.8 |
| MACH-04 | 52.4 |
| MACH-05 | 145.0 |

La valeur `145.0` est très différente des autres températures.

Elle peut être considérée comme un outlier.

## 6. Un outlier n'est pas forcément une erreur

Un outlier peut être une erreur, mais pas toujours.

Il peut être :

- une erreur capteur ;
- une erreur de saisie ;
- une mauvaise unité ;
- une donnée dupliquée ou mal importée ;
- un événement rare mais réel ;
- un signal de panne ou d'anomalie.

Exemple :

Une température très élevée peut être :

- fausse, si le capteur est défaillant ;
- vraie, si la machine a réellement surchauffé.

Il faut donc vérifier le contexte avant de supprimer ou corriger un outlier.

La bonne question n'est pas seulement :

```text
Est-ce que cette valeur est différente ?
```

La bonne question est plutôt :

```text
Est-ce que cette valeur est impossible, ou seulement rare ?
```

## 7. Différence entre classe et outlier

Une classe est une catégorie.

Un outlier est une valeur atypique.

| Notion | Définition | Exemple |
|---|---|---|
| Classe | Catégorie à prédire ou analyser | `panne`, `pas de panne` |
| Outlier | Valeur très éloignée des autres | température `145.0` |

La classe répond à la question :

```text
À quelle catégorie appartient cette ligne ?
```

L'outlier répond à la question :

```text
Cette valeur est-elle anormalement éloignée des autres ?
```

## 8. Exemple combiné

Une même ligne peut avoir une classe et contenir un outlier.

| machine_id | temperature_c | pressure_bar | classe |
|---|---:|---:|---|
| MACH-05 | 145.0 | 180.2 | panne |

Ici :

- `panne` est la classe ;
- `145.0` peut être un outlier sur la température.

Dans un contexte industriel, cet outlier peut être très important, car il peut expliquer la panne.

## 9. À retenir

- Une classe sert à catégoriser une ligne.
- Une classe peut être la valeur que l'on cherche à prédire.
- Un outlier est une valeur très éloignée des autres.
- Un outlier doit être analysé avant d'être supprimé.
- Certains outliers sont des erreurs.
- Certains outliers sont des événements rares mais importants.
- Une classe rare et un outlier sont deux notions différentes, mais elles peuvent être liées.

## Conclusion

Une classe est une catégorie utilisée pour organiser, analyser ou prédire les données.

Un outlier est une valeur atypique qui s'écarte fortement du comportement général.

Ces deux notions sont différentes, mais elles peuvent se croiser : certains outliers peuvent aider à détecter une classe rare, comme une panne ou un incident critique.
