# Dédoublonnage et imputation des données

## Objectif

Avant d'analyser ou d'entraîner un modèle sur un jeu de données, il faut vérifier deux problèmes fréquents :

- les doublons, c'est-à-dire plusieurs lignes qui représentent le même événement ou le même objet ;
- les données manquantes, c'est-à-dire des valeurs absentes dans certaines colonnes.

Ces deux sujets doivent être traités avec prudence, car une mauvaise correction peut créer de fausses informations et dégrader l'analyse.

## 1. Dédoublonnage

Le dédoublonnage consiste à repérer et supprimer les lignes répétées ou quasi répétées.

Dans un dataset industriel, un doublon peut par exemple être :

- deux lignes de télémétrie pour la même machine au même timestamp ;
- deux signalements d'incident avec le même identifiant ;
- deux événements très proches qui décrivent en réalité la même panne.

## 2. Doublons exacts

Un doublon exact est une ligne strictement identique à une autre.

Exemple :

| machine_id | timestamp | temperature_c | pressure_bar |
|---|---:|---:|---:|
| MACH-01 | 2025-06-01 10:00 | 52.1 | 198.4 |
| MACH-01 | 2025-06-01 10:00 | 52.1 | 198.4 |

Dans ce cas, on peut généralement supprimer une des deux lignes.

En pandas :

```python
df = df.drop_duplicates()
```

Si la règle métier dit qu'une machine ne doit avoir qu'une mesure par timestamp :

```python
df = df.drop_duplicates(subset=["machine_id", "timestamp"])
```

## 3. Doublons partiels ou métiers

Un doublon métier n'est pas forcément identique sur toutes les colonnes.

Exemple :

| incident_id | machine_id | date | time | comment |
|---|---|---|---|---|
| INC-001 | MACH-02 | 2025-06-01 | 10:00 | vibration anormale |
| INC-001 | MACH-02 | 2025-06-01 | 10:02 | vibration forte |

Ici, les commentaires et l'heure changent légèrement, mais il peut s'agir du même incident.

Il faut donc définir une clé métier :

- pour la télémétrie : `machine_id + timestamp` ;
- pour les incidents : `incident_id` ;
- pour les machines : `machine_id` ou `machine_code`.

## 4. Dédoublonnage avec agrégation

Quand plusieurs lignes existent pour la même clé, on peut parfois les fusionner au lieu d'en supprimer une arbitrairement.

Exemple pour la télémétrie :

```python
signal_cols = [
    "temperature_c",
    "pressure_bar",
    "voltage_mean_v",
    "rotation_mean_rpm",
    "pieces_produced",
]

telemetry = (
    telemetry
    .groupby(["machine_id", "timestamp"], as_index=False)[signal_cols]
    .mean()
)
```

Ici, les mesures numériques sont moyennées pour obtenir une seule ligne par machine et par heure.

## 5. Dédoublonnage par similarité

Quand les doublons ne sont pas exacts, on peut utiliser des méthodes de similarité.

Exemples :

- distance entre textes de commentaires ;
- comparaison entre noms proches ;
- comparaison entre timestamps proches ;
- regroupement par machine, période et type d'incident.

La méthode KNN peut être utilisée dans ce contexte : elle cherche les lignes les plus proches selon plusieurs variables.

Attention cependant : KNN ne décide pas à lui seul qu'une ligne est un doublon. Il aide seulement à repérer des candidats proches. La règle finale doit rester métier.

## 6. Données manquantes

Une donnée manquante est une valeur absente dans une colonne.

Exemple :

| machine_id | timestamp | temperature_c | pressure_bar |
|---|---:|---:|---:|
| MACH-01 | 2025-06-01 10:00 | 52.1 | 198.4 |
| MACH-01 | 2025-06-01 11:00 |  | 199.1 |

Ici, `temperature_c` est manquante sur la deuxième ligne.

## 7. Diagnostiquer la raison du manque

Avant de remplir une valeur manquante, il faut comprendre pourquoi elle manque.

Questions utiles :

- le capteur était-il hors service ?
- la machine était-elle arrêtée ?
- la donnée a-t-elle été mal importée ?
- la valeur est-elle réellement inexistante ?
- le champ est-il facultatif ?

Cette étape est importante, car toutes les valeurs manquantes ne doivent pas être remplacées de la même façon.

## 8. Supprimer une ligne

On peut supprimer une ligne si elle contient trop de données manquantes.

Exemple :

```python
df = df.dropna(thresh=4)
```

Cette stratégie est simple, mais elle peut faire perdre de l'information.

Elle est pertinente si :

- la ligne est inutilisable ;
- la proportion de lignes supprimées reste faible ;
- la suppression ne crée pas de biais.

## 9. Imputation par constante

L'imputation par constante consiste à remplacer les valeurs manquantes par une valeur fixe.

Exemples :

- `False` pour une colonne booléenne ;
- `0` pour un compteur ;
- `"inconnu"` pour une catégorie ;
- `"non renseigné"` pour un commentaire absent.

Exemple :

```python
df["comment"] = df["comment"].fillna("non renseigné")
df["type_vibration"] = df["type_vibration"].fillna(0)
```

Cette méthode est utile pour les booléens ou les petites séries numériques simples.

## 10. Imputation par règle métier

On peut aussi imputer par programmation avec des règles métier.

Exemple :

- si le commentaire contient `"surchauffe"`, alors `type_surchauffe = 1` ;
- si la criticité machine est élevée et que le commentaire parle d'arrêt, alors l'incident peut être marqué comme critique ;
- si une mesure de pression manque mais que la machine est arrêtée, la valeur manquante peut être conservée plutôt que remplacée.

Exemple simplifié :

```python
df.loc[
    df["comment"].str.contains("surchauffe", case=False, na=False),
    "type_surchauffe",
] = 1
```

Cette stratégie est souvent plus explicable qu'une imputation automatique.

## 11. Imputation par moyenne

L'imputation par moyenne remplace une valeur manquante par la moyenne de la colonne.

Exemple :

```python
df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].mean())
```

Cette méthode peut fonctionner si les données sont bien distribuées et sans valeurs extrêmes importantes.

Elle est moins adaptée si :

- la distribution est très asymétrique ;
- il y a beaucoup de valeurs aberrantes ;
- la moyenne n'a pas de sens métier.

## 12. Imputation par médiane

L'imputation par médiane remplace une valeur manquante par la valeur centrale.

Exemple :

```python
df["pressure_bar"] = df["pressure_bar"].fillna(df["pressure_bar"].median())
```

La médiane est plus robuste que la moyenne quand la distribution est asymétrique ou contient des valeurs extrêmes.

## 13. Imputation intelligente

Une imputation plus avancée peut utiliser plusieurs colonnes pour prédire la valeur manquante.

Exemples de méthodes :

- KNN imputer ;
- `IterativeImputer` ;
- modèle de régression ;
- règles par machine, shift ou période.

Exemple avec `KNNImputer` :

```python
from sklearn.impute import KNNImputer

cols = ["temperature_c", "pressure_bar", "rotation_mean_rpm"]

imputer = KNNImputer(n_neighbors=5)
df[cols] = imputer.fit_transform(df[cols])
```

Cette approche peut être utile, mais elle demande plus de prudence.

## 14. Attention au train-test split

Pour un projet de machine learning, l'imputation doit éviter la fuite de données.

Il ne faut pas calculer la moyenne, la médiane ou l'imputer sur tout le dataset avant de séparer train et test.

Mauvaise pratique :

```python
df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].mean())
train, test = train_test_split(df)
```

Bonne pratique :

```python
train, test = train_test_split(df)

median_temperature = train["temperature_c"].median()
train["temperature_c"] = train["temperature_c"].fillna(median_temperature)
test["temperature_c"] = test["temperature_c"].fillna(median_temperature)
```

Encore mieux : utiliser un pipeline scikit-learn.

## 15. Bonnes pratiques

- Toujours mesurer le nombre de valeurs manquantes avant correction.
- Garder une trace des lignes modifiées.
- Ne pas imputer sans comprendre la cause du manque.
- Préférer une règle simple et explicable si elle suffit.
- Ne pas supprimer trop vite les lignes incomplètes.
- Vérifier l'impact de l'imputation sur les statistiques.
- En machine learning, faire l'imputation après le train-test split.

## 16. Exemple de diagnostic simple

```python
missing_summary = (
    df.isna()
    .sum()
    .rename_axis("colonne")
    .reset_index(name="valeurs_absentes")
    .query("valeurs_absentes > 0")
    .sort_values("valeurs_absentes", ascending=False)
)

display(missing_summary)
```

Pour connaître les lignes incomplètes :

```python
df_incomplete = df[df.isna().any(axis=1)]
```

Pour connaître les colonnes absentes par ligne :

```python
df_incomplete["colonnes_absentes"] = df.isna().apply(
    lambda row: "; ".join(row.index[row]),
    axis=1,
)
```

## Conclusion

Le dédoublonnage et l'imputation sont des étapes de préparation indispensables.

Le bon réflexe est :

1. détecter les doublons et les valeurs manquantes ;
2. comprendre leur origine ;
3. choisir une règle simple, explicable et adaptée au métier ;
4. documenter la correction appliquée ;
5. vérifier que la correction ne crée pas de biais.

Une donnée manquante ou dupliquée n'est pas seulement un problème technique : c'est souvent un signal sur la qualité du processus de collecte.
