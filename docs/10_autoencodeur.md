# Auto-encodeur convolutionnel — détection de défauts visuels

## Objectif

Cette partie met en place une première approche de détection d'anomalies sur des images de bouteilles : un **auto-encodeur convolutionnel** apprend uniquement l'apparence des pièces saines. Lorsqu'une image est mal reconstruite, son erreur de reconstruction sert de score d'anomalie.

L'objectif est donc différent d'une classification supervisée : les images défectueuses ne sont pas utilisées pour apprendre les paramètres du réseau.

## Jeu de données et préparation

Le jeu employé est la catégorie `bottle` de MVTec AD. Les images sont versionnées à partir de leur contenu : un manifeste contient les chemins, les empreintes SHA-256, les séparations et les masques de vérité terrain. Une même image retrouvée dans deux jeux provoque l'arrêt de la préparation, afin d'éviter une fuite évidente entre entraînement et évaluation.

Les images sont :

- converties en RGB ;
- redimensionnées à `256 × 256` sans déformer la bouteille : l'image est contenue dans un carré puis complétée par un fond noir (*letterbox padding*) ;
- converties en `float32` et ramenées dans `[0, 1]` ;
- accompagnées, pour les défauts, de masques pixel à pixel utilisés uniquement pour l'évaluation de la localisation.

Le découpage est déterministe (graine `42`) : 20 % des images `train/good` sont réservées à la validation. Les images anormales issues du dossier de test sont aussi séparées entre validation et test, mais elles ne participent ni à l'entraînement ni au réglage du seuil de la présente exécution.

Des augmentations faibles sont appliquées à la volée **uniquement** aux images saines d'entraînement : retournement horizontal avec probabilité 0,5, rotation entre -5° et +5°, translation maximale de 3 %, luminosité et contraste entre 0,9 et 1,1. Le retournement vertical est volontairement exclu, car il ne correspond pas à l'orientation physique attendue d'une bouteille.

## Principe de l'auto-encodeur

Le modèle reçoit une image `x` et produit une reconstruction `x̂`.

```text
image saine x → encodeur → représentation latente z → décodeur → reconstruction x̂
                                      │
                         goulot d'information
```

En théorie, le réseau apprend bien les motifs habituels des bouteilles saines. Une rayure, une contamination ou une cassure, peu présentes dans cet apprentissage, devraient être moins bien reconstruites. Le score utilisé est la MSE moyenne des pixels :

```text
score(x) = moyenne((x - x̂)²)
```

Un score supérieur au seuil correspond à une alerte. La même erreur, moyenne sur les trois canaux RGB mais conservée à chaque pixel, produit une carte thermique pour localiser les zones suspectes.

## Architecture entraînée

L'architecture est volontairement compacte : trois convolutions pour encoder et trois convolutions transposées pour décoder. Les activations intermédiaires sont ReLU ; la dernière couche utilise une sigmoïde pour produire des pixels dans `[0, 1]`.

| Étape | Sortie | Rôle |
|---|---:|---|
| Entrée | `256 × 256 × 3` | image RGB normalisée |
| Conv2D, 32 filtres, stride 2 | `128 × 128 × 32` | premier niveau de caractéristiques |
| Conv2D, 64 filtres, stride 2 | `64 × 64 × 64` | compression spatiale |
| Conv2D, 16 filtres, stride 2 | `32 × 32 × 16` | espace latent |
| Conv2DTranspose, 64 filtres, stride 2 | `64 × 64 × 64` | décodage |
| Conv2DTranspose, 32 filtres, stride 2 | `128 × 128 × 32` | décodage |
| Conv2DTranspose, 3 filtres, stride 2 | `256 × 256 × 3` | reconstruction RGB |

Le modèle comporte **57 235 paramètres**. L'entrée contient 196 608 valeurs et l'espace latent 16 384, soit un rapport de compression de **12**. Ce goulot cherche à empêcher le réseau de simplement copier l'image ; il ne garantit toutefois pas à lui seul une bonne séparation entre défauts et pièces saines.

## Entraînement réalisé

L'entraînement est effectué avec Keras 3 et le backend PyTorch. La graine aléatoire est fixée à 42. L'optimiseur est Adam et la fonction de perte est la MSE de reconstruction. Un `EarlyStopping` surveille la perte des images saines de validation, avec une patience de 5 époques et restauration des meilleurs poids.

| Paramètre | Valeur du run |
|---|---:|
| Images saines d'entraînement | 167 |
| Images saines de validation | 42 |
| Images de test | 64 |
| Époques demandées / réalisées | 20 / 20 |
| Taille de lot | 8 |
| Taux d'apprentissage | 0,001 |
| Perte | MSE |
| Meilleure époque | 20 |
| Meilleure perte de validation | 0,001164 |

Le seuil n'est pas ajusté sur les défauts : il est fixé au **99e centile** des 42 scores de validation sains, soit `0,001321`. Cela traduit une hypothèse simple : environ 1 % des pièces saines comparables à la validation pourraient dépasser ce seuil.

Les artefacts produits sont le modèle `.keras`, son résumé, la courbe d'apprentissage, l'histogramme des scores, la matrice de confusion, des reconstructions, des heatmaps et un rapport JSON. Les paramètres, métriques et artefacts sont également suivis avec MLflow.

## Résultats obtenus et interprétation

| Mesure sur le test final | Résultat |
|---|---:|
| AUROC image | 0,477 |
| Average Precision image | 0,734 |
| Précision au seuil | 0,733 |
| Rappel au seuil | 0,250 |
| F1-score | 0,373 |
| Vrais négatifs / faux positifs | 16 / 4 |
| Faux négatifs / vrais positifs | 33 / 11 |
| AUROC pixel | 0,563 |
| Average Precision pixel | 0,081 |

Le résultat principal est insuffisant : un AUROC de 0,477 est inférieur au comportement attendu d'un classement aléatoire (environ 0,5). Les scores moyens des images saines et défectueuses sont presque identiques (`0,001242` contre `0,001246`). Le modèle ne sépare donc pratiquement pas les deux populations.

Le seuil de 99e centile a l'avantage d'être calibré sans regarder le test ni les défauts. En revanche, ici il ne détecte que 11 défauts sur 44 et en manque 33. Les quatre fausses alertes sont également à considérer dans un contexte industriel. La précision semble correcte, mais elle ne compense pas un rappel de 25 % : trois défauts sur quatre ne déclenchent pas d'alerte.

Les cartes thermiques doivent être comparées aux masques : une forte erreur sur le contour, le fond noir ou un effet de redimensionnement peut augmenter le score sans correspondre au défaut réel. L'AUROC pixel de 0,563 et l'Average Precision pixel de 0,081 confirment que la localisation est faible.

## Limites de l'approche actuelle

1. **Peu de données pour apprendre la variabilité normale.** Les 167 images saines sont peu nombreuses pour modéliser précisément éclairage, pose, reflets et variations de fabrication.
2. **Architecture de référence très simple.** Sans connexions de saut, normalisation, attention ni encodeur préentraîné, le réseau peut perdre des détails utiles ou, au contraire, reconstruire une partie des défauts.
3. **MSE peu perceptuelle.** Elle pénalise uniformément les écarts pixel à pixel et est sensible aux petites variations d'alignement ou de luminosité. Elle ne correspond pas toujours à la perception d'un défaut métier.
4. **Seuil fondé sur seulement 42 images.** Le 99e centile est instable sur un échantillon réduit : il dépend fortement de quelques images et ne traduit pas directement le coût métier d'un défaut manqué.
5. **Validation incomplètement exploitée.** Une partie des défauts est réservée à la validation par la préparation du jeu, mais le pipeline d'entraînement n'utilise que les images saines de validation. Cette décision protège contre l'optimisation sur les défauts, mais laisse les données de validation anormales sans rôle pour comparer des configurations ou choisir un compromis opérationnel.
6. **Contrôle de fuite limité.** SHA-256 détecte les doublons exacts, pas deux prises de vue très proches de la même bouteille, d'un même lot ou de la même série. MVTec est aussi un jeu académique, différent d'une ligne industrielle réelle.
7. **Une seule exécution.** La graine est fixe et il n'y a ni réplication sur plusieurs graines ni intervalle de confiance. Le résultat peut varier avec le découpage ou l'initialisation.
8. **Risque de biais de fond.** Le padding noir, les transformations avec remplissage noir et les contours de la bouteille peuvent concentrer l'erreur de reconstruction et perturber le score.

## Améliorations prioritaires

1. **Établir une vraie comparaison expérimentale.** Tester plusieurs graines et conserver un test final strictement intouchable. Comparer MSE, SSIM et une perte combinée (par exemple MSE + SSIM), ainsi que plusieurs tailles de goulot et taux d'apprentissage.
2. **Utiliser la validation de manière cohérente.** Garder les défauts de validation pour sélectionner l'architecture et analyser les courbes précision-rappel ou le rappel à taux de fausses alertes fixé. Le test final ne doit être consulté qu'une fois les choix arrêtés. Si l'objectif est strictement non supervisé, déclarer explicitement ce choix et ne pas créer de validation anormale inutilisée.
3. **Choisir le seuil selon le risque métier.** Au lieu d'un centile arbitraire, fixer par exemple le seuil qui garantit un rappel minimal, ou un nombre maximal de fausses alertes par poste. Présenter la courbe précision-rappel et l'impact des faux négatifs.
4. **Améliorer la représentation.** Essayer un auto-encodeur U-Net, un encodeur préentraîné sur des images générales ou industrielles, ou des méthodes dédiées à MVTec AD telles que PatchCore, PaDiM ou EfficientAD. Elles travaillent souvent sur des caractéristiques locales et sont mieux adaptées aux petits défauts.
5. **Réduire l'effet du fond.** Segmenter ou masquer la bouteille avant le calcul du score, employer un remplissage plus neutre ou recadrer autour de l'objet. Évaluer séparément l'erreur à l'intérieur et à l'extérieur de la région d'intérêt.
6. **Renforcer l'évaluation.** Rapporter AUROC et Average Precision image/pixel par type de défaut, rappel à un taux de fausses alertes donné, et analyser visuellement les faux négatifs. Les masques doivent servir à quantifier la localisation, pas uniquement à illustrer des heatmaps.
7. **Se rapprocher des données réelles.** Constituer un jeu par lots et par caméras, séparer les séries entre train/validation/test, documenter les défauts ambigus avec un expert qualité et surveiller la dérive après déploiement.

## Alternative disponible : PatchCore

L'interface propose désormais **PatchCore (ResNet-18)** en plus de l'auto-encodeur. Contrairement à ce dernier, PatchCore ne reconstruit pas l'image et n'utilise ni époques, ni taux d'apprentissage, ni fonction de perte. Il extrait des caractéristiques locales avec un ResNet-18 pré-entraîné sur ImageNet, puis conserve une banque représentative de patchs issus des images saines. Le score d'un patch est sa distance au patch sain le plus proche ; le score d'une image est le maximum de ces distances.

Les paramètres spécifiques sont le ratio de coreset, le nombre de patchs candidats et le nombre maximal de patchs conservés en mémoire. Des valeurs plus grandes peuvent améliorer la couverture des variations normales, mais augmentent fortement le temps de sélection du coreset et le coût de calcul des plus proches voisins. Le premier lancement télécharge les poids ResNet-18 ; les lancements suivants réutilisent le cache local.

PatchCore est introduit comme une comparaison plus adaptée aux défauts locaux que l'auto-encodeur simple. Il faut néanmoins conserver exactement le même protocole : apprentissage sur les seules images saines, seuil calibré sur la validation saine et test final jamais utilisé pour le réglage.

## Conclusion

Le pipeline est reproductible, traçable et méthodologiquement prudent sur deux points importants : l'apprentissage n'utilise que des images saines et le seuil est calibré sans consulter le test final. Il fournit donc une bonne base pédagogique et technique pour explorer la détection d'anomalies.

En revanche, les métriques du run montrent clairement que cette version ne doit pas être utilisée pour une décision qualité industrielle : elle manque 75 % des défauts et ne les classe pas mieux que le hasard. La prochaine étape pertinente est une étude comparative contrôlée — incluant un modèle à caractéristiques préentraînées et une calibration de seuil orientée métier — avant toute conclusion sur la faisabilité.

## Traçabilité du run documenté

- Run : `vision_ae_20260721145036_7c02ba1d`
- Rapport : `backend/artifacts/vision-model-runs/vision_ae_20260721145036_7c02ba1d/report.json`
- Code : `backend/src/indusense/vision_dataset.py`, `backend/src/indusense/vision/model.py`, `backend/src/indusense/vision/train.py` et `backend/src/indusense/vision/anomaly.py`
