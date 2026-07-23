# Régressions

## Objectif

Cette fiche résume les idées principales du support `03_Regressions.pdf` : régression linéaire, descente de gradient, régularisation, régression polynomiale et régression logistique.

Une régression sert à prédire une valeur numérique ou une probabilité à partir de variables explicatives.

## 1. Régression linéaire

La régression linéaire cherche une relation simple entre les variables d'entrée et la cible.

Forme générale :

```text
y = w1*x1 + w2*x2 + ... + b
```

Le modèle apprend les poids `w` et le biais `b` pour minimiser l'erreur entre les prédictions et les vraies valeurs.

Exemples d'usage :

- prédire une température ;
- estimer une consommation électrique ;
- prévoir un coût de maintenance ;
- estimer une durée avant panne.

## 2. Descente de gradient

La descente de gradient est une méthode d'optimisation.

Elle ajuste progressivement les paramètres du modèle pour réduire la fonction de coût.

Idée simple :

```text
paramètres -> prédictions -> erreur -> correction des paramètres
```

Le learning rate contrôle la taille des corrections :

- trop faible : apprentissage lent ;
- trop élevé : apprentissage instable ;
- bien choisi : convergence progressive.

## 3. Normalisation et standardisation

Les variables n'ont pas toujours la même échelle.

Exemple :

| variable | ordre de grandeur |
|---|---:|
| température | 20 à 120 |
| pression | 1 à 300 |
| vibration | 0 à 1 |

La standardisation met souvent les variables autour d'une moyenne de `0` et d'un écart-type de `1`.

C'est important pour les modèles sensibles aux échelles, notamment ceux optimisés par descente de gradient.

## 4. Régularisation

La régularisation limite les coefficients trop grands.

Elle aide à réduire le surapprentissage.

Principales formes :

| méthode | idée |
|---|---|
| L1 / Lasso | peut ramener certains coefficients à zéro |
| L2 / Ridge | pénalise les grands coefficients |
| Elastic Net | combine L1 et L2 |

## 5. Régression polynomiale

La régression polynomiale ajoute des termes non linéaires.

Exemple :

```text
y = a*x² + b*x + c
```

Elle permet de représenter des courbes, mais augmente le risque de surapprentissage si le degré est trop élevé.

## 6. Régression logistique

La régression logistique est utilisée pour la classification.

Elle produit une probabilité entre `0` et `1` grâce à la fonction sigmoïde.

```text
probabilité >= seuil -> classe positive
probabilité < seuil  -> classe négative
```

Exemple :

```text
P(panne dans les 24h) = 0.82 -> panne probable
```

Pour l'entraînement, on utilise souvent la logloss plutôt qu'une erreur quadratique.

## À retenir

- Régression linéaire : prédire une valeur continue.
- Régression logistique : prédire une probabilité de classe.
- Standardiser aide les modèles sensibles aux échelles.
- Régulariser limite le surapprentissage.
- Les modèles plus flexibles ne sont pas toujours meilleurs.
