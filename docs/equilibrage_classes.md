# Équilibrage des classes

## Objectif

En machine learning, on parle de classes quand on cherche à prédire une catégorie.

Exemples :

- panne ou pas panne ;
- incident critique ou incident non critique ;
- défaut qualité ou fonctionnement normal ;
- type d'incident : vibration, surchauffe, baisse de pression, etc.

Un dataset est déséquilibré quand une classe est beaucoup plus fréquente que les autres.

## 1. Exemple simple

Imaginons un dataset d'incidents industriels :

| classe | nombre de lignes |
|---|---:|
| Pas de panne | 9 500 |
| Panne | 500 |

Ici, la classe `Pas de panne` représente 95 % des données.

La classe `Panne` représente seulement 5 %.

Le dataset est donc déséquilibré.

## 2. Pourquoi c'est un problème ?

Un modèle peut obtenir un bon score global sans vraiment apprendre la classe minoritaire.

Exemple :

Si 95 % des lignes sont `Pas de panne`, un modèle qui prédit toujours `Pas de panne` aura déjà 95 % d'accuracy.

Mais ce modèle est inutile, car il ne détecte aucune panne.

Dans ce cas, l'accuracy est trompeuse.

## 3. Classe majoritaire et classe minoritaire

La classe majoritaire est la classe la plus fréquente.

La classe minoritaire est la classe la moins fréquente.

Exemple :

| rôle | classe |
|---|---|
| Classe majoritaire | Pas de panne |
| Classe minoritaire | Panne |

Dans un contexte industriel, la classe minoritaire est souvent la plus importante, car elle correspond aux événements rares :

- panne ;
- arrêt urgence ;
- incident critique ;
- défaut qualité grave ;
- anomalie capteur.

## 4. Diagnostiquer le déséquilibre

Avant de corriger un déséquilibre, il faut le mesurer.

```python
target_col = "is_breakdown"

class_counts = (
    df[target_col]
    .value_counts()
    .rename_axis("classe")
    .reset_index(name="lignes")
)

class_counts["part"] = class_counts["lignes"] / class_counts["lignes"].sum()

display(class_counts)
```

On peut aussi visualiser la répartition :

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
sns.barplot(data=class_counts, x="classe", y="lignes")
plt.title("Répartition des classes")
plt.xlabel("Classe")
plt.ylabel("Nombre de lignes")
plt.tight_layout()
plt.show()
```

## 5. Downsampling

Le downsampling consiste à réduire artificiellement la classe majoritaire.

Exemple :

| classe | avant | après downsampling |
|---|---:|---:|
| Pas de panne | 9 500 | 500 |
| Panne | 500 | 500 |

On supprime donc une partie des exemples de la classe majoritaire pour obtenir un dataset équilibré.

## 6. Exemple de downsampling

```python
from sklearn.utils import resample

majority = df[df["is_breakdown"] == False]
minority = df[df["is_breakdown"] == True]

majority_downsampled = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42,
)

df_balanced = pd.concat([majority_downsampled, minority])
```

Avantage :

- méthode simple ;
- rapide ;
- limite le risque de créer de fausses données.

Inconvénient :

- on perd des informations de la classe majoritaire ;
- le modèle peut moins bien apprendre les cas normaux.

## 7. Oversampling

L'oversampling consiste à augmenter artificiellement la classe minoritaire.

Exemple :

| classe | avant | après oversampling |
|---|---:|---:|
| Pas de panne | 9 500 | 9 500 |
| Panne | 500 | 9 500 |

L'idée est de donner plus de poids à la classe rare.

## 8. Oversampling par duplication

La méthode la plus simple consiste à dupliquer des lignes de la classe minoritaire.

```python
minority_oversampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42,
)

df_balanced = pd.concat([majority, minority_oversampled])
```

Cette méthode est simple, mais elle a un risque important : le modèle peut apprendre par coeur les exemples dupliqués.

On parle alors de surapprentissage.

## 9. SMOTE

SMOTE signifie `Synthetic Minority Oversampling Technique`.

Contrairement à une duplication simple, SMOTE génère de nouveaux exemples synthétiques pour la classe minoritaire.

Il ne recopie pas directement les pannes réelles.

Il crée de nouveaux cas à partir des plus proches voisins de la classe minoritaire.

L'idée est :

1. prendre un exemple minoritaire ;
2. trouver ses voisins proches ;
3. créer un nouveau point entre cet exemple et un voisin ;
4. répéter jusqu'à équilibrer les classes.

## 10. Exemple avec SMOTE

```python
from imblearn.over_sampling import SMOTE

X = df.drop(columns=["is_breakdown"])
y = df["is_breakdown"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

Après SMOTE, les classes sont plus équilibrées.

```python
y_resampled.value_counts()
```

## 11. Attention avec SMOTE

SMOTE ne doit pas être utilisé n'importe comment.

Il fonctionne surtout avec des variables numériques.

Il peut être dangereux si :

- les variables catégorielles ne sont pas correctement encodées ;
- les données synthétiques n'ont pas de sens métier ;
- les classes se chevauchent beaucoup ;
- le dataset contient des outliers ;
- on l'applique avant le train-test split.

## 12. Attention au train-test split

Il faut faire le split avant l'oversampling.

Mauvaise pratique :

```python
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
```

Bonne pratique :

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

Le test set doit rester représentatif de la réalité.

Si on applique SMOTE avant le split, des informations synthétiques proches peuvent se retrouver à la fois dans le train et dans le test.

C'est une fuite de données.

## 13. Stratification

Quand les classes sont déséquilibrées, il faut souvent utiliser `stratify`.

Cela conserve la proportion des classes dans le train et dans le test.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

Sans stratification, il est possible d'avoir trop peu d'exemples minoritaires dans le test set.

## 14. Métriques adaptées

Avec un dataset déséquilibré, il ne faut pas regarder uniquement l'accuracy.

Métriques plus utiles :

- recall ;
- precision ;
- F1-score ;
- matrice de confusion ;
- ROC-AUC ;
- PR-AUC.

Pour la détection de panne, le recall est souvent très important.

Il mesure la capacité du modèle à retrouver les vraies pannes.

```text
recall = pannes détectées / total des vraies pannes
```

## 15. Pondération des classes

Une autre solution consiste à donner plus de poids à la classe minoritaire.

Certains modèles acceptent un paramètre `class_weight`.

Exemple :

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
)
```

Cette méthode ne modifie pas le dataset.

Elle indique simplement au modèle que les erreurs sur la classe rare coûtent plus cher.

## 16. Choisir la bonne stratégie

| Situation | Stratégie possible |
|---|---|
| Dataset très grand | Downsampling possible |
| Peu de cas minoritaires | Oversampling ou SMOTE |
| Données numériques propres | SMOTE possible |
| Données catégorielles nombreuses | Prudence avec SMOTE |
| Besoin d'explicabilité | Pondération ou downsampling |
| Risque métier fort sur la classe rare | Optimiser recall ou F1-score |

## 17. Bonnes pratiques

- Mesurer le déséquilibre avant de corriger.
- Garder un test set réaliste.
- Appliquer SMOTE seulement sur le train set.
- Ne pas dupliquer aveuglément les pannes réelles.
- Vérifier que les exemples synthétiques ont un sens métier.
- Utiliser des métriques adaptées.
- Comparer plusieurs stratégies.
- Documenter la méthode retenue.

## Conclusion

Un dataset est déséquilibré quand une classe est beaucoup plus fréquente que les autres.

Pour corriger cela, on peut :

- réduire la classe majoritaire avec du downsampling ;
- augmenter la classe minoritaire avec de l'oversampling ;
- générer des exemples synthétiques avec SMOTE ;
- pondérer les classes dans le modèle.

Le point le plus important : ne jamais équilibrer le test set.

Le test set doit rester proche de la réalité pour mesurer correctement les performances du modèle.
