# Outliers et Z-score

## Objectif

Dans un jeu de données, certaines valeurs peuvent être très éloignées du comportement général.

Ces valeurs sont appelées :

- valeurs atypiques ;
- valeurs extrêmes ;
- outliers.

Un outlier n'est pas automatiquement une erreur. Il peut être :

- une vraie situation rare ;
- une panne importante ;
- une mesure capteur anormale ;
- une erreur de saisie ;
- un problème d'import ;
- une donnée impossible physiquement.

L'objectif n'est donc pas de supprimer automatiquement les outliers, mais de les détecter, les comprendre et décider quoi faire.

## 1. Définition

Un outlier est une donnée qui s'écarte fortement du comportement général des autres données.

Exemple simple :

| machine_id | temperature_c |
|---|---:|
| MACH-01 | 51.2 |
| MACH-02 | 49.8 |
| MACH-03 | 50.4 |
| MACH-04 | 52.1 |
| MACH-05 | 145.0 |

Ici, `145.0` est très éloigné des autres températures. C'est une valeur suspecte.

Elle peut être :

- vraie, si la machine a réellement surchauffé ;
- fausse, si le capteur ou l'import a produit une erreur.

## 2. Pourquoi détecter les outliers ?

Les outliers peuvent fortement perturber les analyses.

Ils peuvent modifier :

- la moyenne ;
- l'écart-type ;
- les corrélations ;
- les graphiques ;
- les modèles de machine learning ;
- les conclusions métier.

Exemple :

```text
Valeurs : 50, 51, 49, 52, 145
```

La moyenne devient plus élevée à cause de `145`, alors que la majorité des valeurs est autour de `50`.

## 3. Outlier vrai ou erreur ?

La première question à poser est toujours :

> Est-ce que cette valeur est impossible, ou seulement rare ?

Exemples :

| Cas | Interprétation possible |
|---|---|
| Température très élevée | vraie surchauffe ou erreur capteur |
| Pression négative | souvent impossible physiquement |
| Production de pièces très élevée | pic réel ou doublon |
| Vibration extrême | panne mécanique possible |
| Timestamp incohérent | erreur d'import ou fuseau horaire |

Une valeur atypique doit donc être analysée avec le contexte métier.

## 4. Méthode du Z-score

Le Z-score mesure l'écart d'une valeur par rapport à la moyenne, en nombre d'écarts-types.

Formule :

```text
z = (valeur - moyenne) / écart-type
```

Interprétation :

- `z = 0` : la valeur est égale à la moyenne ;
- `z = 1` : la valeur est à 1 écart-type au-dessus de la moyenne ;
- `z = -1` : la valeur est à 1 écart-type en dessous de la moyenne ;
- `z = 3` : la valeur est très éloignée de la moyenne.

On considère souvent une valeur comme suspecte si :

```text
|z| > 3
```

Cela signifie que la valeur est à plus de 3 écarts-types de la moyenne.

## 5. Exemple simple

```python
import pandas as pd

df = pd.DataFrame({
    "temperature_c": [50, 51, 49, 52, 145]
})

mean = df["temperature_c"].mean()
std = df["temperature_c"].std()

df["z_score_temperature"] = (df["temperature_c"] - mean) / std
df["temperature_suspecte"] = df["z_score_temperature"].abs() > 3

display(df)
```

Dans cet exemple, la colonne `temperature_suspecte` indique si la valeur est atypique selon la règle du Z-score.

## 6. Application à plusieurs colonnes

Dans un dataset de télémétrie industrielle, on peut appliquer le Z-score aux signaux numériques :

- `temperature_c` ;
- `pressure_bar` ;
- `voltage_mean_v` ;
- `rotation_mean_rpm` ;
- `pieces_produced`.

Exemple :

```python
signal_cols = [
    "temperature_c",
    "pressure_bar",
    "voltage_mean_v",
    "rotation_mean_rpm",
    "pieces_produced",
]

for col in signal_cols:
    mean = telemetry[col].mean()
    std = telemetry[col].std()
    telemetry[f"z_score_{col}"] = (telemetry[col] - mean) / std
    telemetry[f"{col}_suspect"] = telemetry[f"z_score_{col}"].abs() > 3
```

## 7. Compter les valeurs suspectes

Après avoir calculé les Z-scores, on peut compter les valeurs suspectes par colonne.

```python
outlier_summary = []

for col in signal_cols:
    suspect_col = f"{col}_suspect"
    outlier_summary.append({
        "signal": col,
        "valeurs_suspectes": int(telemetry[suspect_col].sum()),
        "part_suspecte": telemetry[suspect_col].mean(),
    })

outlier_summary = pd.DataFrame(outlier_summary)
display(outlier_summary)
```

Ce tableau permet d'identifier les signaux les plus instables.

## 8. Visualiser les outliers

Les graphiques utiles sont :

- boxplot ;
- histogramme ;
- nuage de points ;
- courbe temporelle ;
- heatmap par machine et signal.

Exemple avec un boxplot :

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.boxplot(data=telemetry[signal_cols])
plt.title("Distribution des signaux de télémétrie")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Le boxplot permet de voir rapidement les valeurs extrêmes.

## 9. Analyser les outliers par machine

Dans un contexte industriel, il est souvent plus utile d'analyser les outliers par machine.

Une valeur extrême peut être normale pour une machine très sollicitée, mais anormale pour une autre.

Exemple :

```python
outlier_by_machine = (
    telemetry
    .groupby("machine_id")
    [[f"{col}_suspect" for col in signal_cols]]
    .sum()
    .reset_index()
)

display(outlier_by_machine)
```

Cette analyse permet de repérer les machines qui produisent le plus de mesures suspectes.

## 10. Limites du Z-score

Le Z-score est simple, mais il a des limites.

Il fonctionne bien si :

- la distribution est relativement proche d'une distribution normale ;
- la moyenne représente bien les données ;
- l'écart-type n'est pas trop influencé par les valeurs extrêmes.

Il fonctionne moins bien si :

- la distribution est très asymétrique ;
- il y a déjà beaucoup d'outliers ;
- les données contiennent plusieurs groupes différents ;
- les valeurs changent selon la machine, le shift ou la période.

Dans ces cas, la moyenne et l'écart-type peuvent être trompeurs.

## 11. Alternatives au Z-score

Si le Z-score n'est pas adapté, on peut utiliser d'autres méthodes.

Les deux alternatives les plus utiles à connaître sont :

- l'IQR, qui s'appuie sur les quartiles ;
- le Z-score robuste, qui s'appuie sur la médiane et le MAD.

## 12. Méthode IQR

IQR signifie `Interquartile Range`, ou intervalle interquartile.

Cette méthode découpe les données en quartiles :

- `Q1` : premier quartile, 25 % des valeurs sont en dessous ;
- `Q2` : médiane, 50 % des valeurs sont en dessous ;
- `Q3` : troisième quartile, 75 % des valeurs sont en dessous.

L'IQR mesure la largeur de la zone centrale des données :

```text
IQR = Q3 - Q1
```

Une valeur est souvent considérée comme suspecte si elle sort de l'intervalle suivant :

```text
borne basse = Q1 - 1.5 * IQR
borne haute = Q3 + 1.5 * IQR
```

Donc une valeur est atypique si :

```text
valeur < borne basse
valeur > borne haute
```

Cette méthode est plus robuste que le Z-score quand la distribution est asymétrique, car elle utilise les quartiles plutôt que la moyenne et l'écart-type.

Exemple :

```python
col = "temperature_c"

q1 = telemetry[col].quantile(0.25)
q3 = telemetry[col].quantile(0.75)
iqr = q3 - q1

borne_basse = q1 - 1.5 * iqr
borne_haute = q3 + 1.5 * iqr

telemetry[f"{col}_outlier_iqr"] = (
    (telemetry[col] < borne_basse)
    | (telemetry[col] > borne_haute)
)
```

## 13. Z-score robuste avec médiane et MAD

Le Z-score classique utilise :

- la moyenne ;
- l'écart-type.

Le problème est que ces deux indicateurs peuvent être fortement influencés par les outliers.

Le Z-score robuste remplace donc :

- la moyenne par la médiane ;
- l'écart-type par le MAD.

MAD signifie `Median Absolute Deviation`, ou écart absolu médian.

Formule simplifiée :

```text
MAD = médiane(|valeur - médiane|)
```

Le Z-score robuste peut ensuite être calculé ainsi :

```text
robust_z = 0.6745 * (valeur - médiane) / MAD
```

On considère souvent une valeur comme suspecte si :

```text
|robust_z| > 3.5
```

Exemple :

```python
col = "temperature_c"

median = telemetry[col].median()
mad = (telemetry[col] - median).abs().median()

telemetry[f"{col}_robust_z_score"] = (
    0.6745 * (telemetry[col] - median) / mad
)
telemetry[f"{col}_outlier_robust_z"] = (
    telemetry[f"{col}_robust_z_score"].abs() > 3.5
)
```

Cette méthode est intéressante quand :

- la distribution n'est pas normale ;
- la distribution est asymétrique ;
- il y a déjà des valeurs extrêmes ;
- on veut une méthode simple mais plus robuste que le Z-score classique.

Il faut toutefois faire attention si `MAD = 0`, car la division devient impossible. Dans ce cas, il vaut mieux utiliser une autre méthode ou traiter la colonne séparément.

## 14. Comparer les méthodes

Une bonne pratique consiste à comparer plusieurs méthodes avant de décider.

```python
outlier_compare = telemetry[[
    "machine_id",
    "timestamp",
    "temperature_c",
    "temperature_c_outlier_iqr",
    "temperature_c_outlier_robust_z",
]]

display(outlier_compare[
    outlier_compare["temperature_c_outlier_iqr"]
    | outlier_compare["temperature_c_outlier_robust_z"]
])
```

Si les deux méthodes détectent la même valeur, elle mérite une vérification prioritaire.

Si une seule méthode la détecte, il faut regarder la distribution et le contexte métier avant de décider.

### Isolation Forest

`IsolationForest` est une méthode de machine learning qui détecte les observations atypiques selon plusieurs variables.

Elle peut être utile quand un outlier dépend d'une combinaison de signaux.

## 15. Que faire avec un outlier ?

Après détection, plusieurs décisions sont possibles.

### Conserver la valeur

Si l'outlier représente un événement réel, il faut le conserver.

Exemple :

- surchauffe réelle ;
- vibration extrême avant panne ;
- chute brutale de pression.

### Corriger la valeur

Si la valeur est une erreur identifiable, on peut la corriger.

Exemple :

- unité mal convertie ;
- virgule mal placée ;
- timestamp décalé.

### Remplacer la valeur

Si la valeur est fausse mais impossible à corriger, on peut l'imputer.

Exemples :

- remplacer par la médiane ;
- remplacer par une valeur calculée par machine ;
- utiliser une méthode d'imputation.

### Supprimer la ligne

Si la ligne est inutilisable, on peut la supprimer.

Cette décision doit rester prudente, surtout si les outliers correspondent à des pannes importantes.

## 16. Bonnes pratiques

- Ne jamais supprimer automatiquement tous les outliers.
- Toujours vérifier si l'outlier peut être une vraie information métier.
- Comparer les valeurs suspectes avec les incidents connus.
- Faire l'analyse par machine si les comportements sont différents.
- Documenter la règle utilisée.
- Garder une colonne de flag plutôt que modifier directement la valeur.
- Vérifier l'impact sur les statistiques avant et après correction.

## 17. Exemple de flag propre

Au lieu de supprimer les valeurs suspectes, on peut ajouter une colonne de contrôle.

```python
col = "temperature_c"

mean = telemetry[col].mean()
std = telemetry[col].std()

telemetry["temperature_z_score"] = (telemetry[col] - mean) / std
telemetry["temperature_outlier_zscore"] = telemetry["temperature_z_score"].abs() > 3
```

Cela permet de conserver la donnée brute tout en indiquant qu'elle mérite une vérification.

## Conclusion

Les outliers sont des valeurs qui s'écartent fortement du comportement général des données.

Le Z-score est une méthode simple pour les détecter :

```text
|z| > 3
```

Mais un outlier n'est pas forcément une erreur. Dans un contexte industriel, il peut être exactement ce qu'on cherche à détecter : une panne, une dérive ou un comportement anormal.

Le bon réflexe est donc :

1. détecter ;
2. comprendre ;
3. contextualiser ;
4. décider ;
5. documenter.
