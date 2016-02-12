# MDI343-Challenge
Challenge du cours MDI343 du Mastère Big Data de Telecom.

***Objectif :*** Prédire l'issue d'une demande de brevet déposée auprès de l'Organisme Européen des Brevets (OEB) : délivrance, ou non-délivrance (par rejet de la demande ou abandon du déposant).

## Soumission finale
```
pip install -r requirements.txt
python final_pred.py
```

## Organisation
Ou comment j'ai travaillé ce challenge ?
### 1. Quick & Dirty
Première étape dans chacun des problèmes de Machine Learning que j'aborde : faire un premier modèle bout à bout en quick & dirty.
Il s'agit d'un modèle rapide qui servira de benchmark pour la suite.

Ce modèle se trouve dans le fichier quick_and_dirty.py et peut être décrit de la manière suivante :
* Preprocessing :
    * Extraction de l'année de toutes les dates.
    * Label encoding de toutes les variables non numériques.
    * Remplacement des valeurs nulles des variables numériques par la moyenne.
* [XGBoost](https://xgboost.readthedocs.org/en/latest/) (eXtreme gradient boosting)
    * Modèle basé sur du gradient boosting
    * Implémentation rapide et efficace grâce à un compilateur C et du multi-threading.

Le résultat de ce modèle quick & dirty permet d'obtenir un score AUC de 0.7147 sur le leaderboard public.


### 2. Preprocessing
Maintenant que notre benchmark est posée, le vrai travail peut commencer.

Premières constations, le problème paraît plutôt 'classique' : classification binaire, 200k/300k données et environ 50 variables.
Sa difficulté paraît être sur son grand nombre de variables catégorielles et de certaines variables qui ont beaucoup de valeurs nulles.

Tout le preprocessing est présent dans le fichier preprocessing.py. Celui-ci a été séparé en plusieurs parties :

1. Remplissage personnalisé de valeurs nulles
2. Travail sur les dates :
    * Extraction du mois / année
    * Différence (timestamp) entre plusieurs dates
3. Créations de nouvelles features : flag (0/1) ou nouvelles variables catégorielles
4. Utilisation de [RFECV](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) pour sélectionner les variables

### 3. Optimisation du modèle
XGBoost a l'air de bien marcher, gardons le et cherchons à l'optimiser.

N'étant pas un grand fan des méthodes RandomizedSearch et GridSearch, je préfère utiliser des librairies d'optimisation bayésienne. J'ai choisi ici d'utiliser hyperopt ([tutorial](https://districtdatalabs.silvrback.com/parameter-tuning-with-hyperopt)).

Afin d'évaluer certains paramètre donnés, j'utilise classiquement 5 fold et sur chaque itération, j'entraîne le modèle sur 80% des données et je l'évalue sur les 20% restants. Je prends la moyenne des scores comme score final d'évaluation.

Dans ce challenge, j'ai utilisé une petite variante. Je suis parti de l'idée que les données étaient en fait séparées en 2 sur la première variable VOIE_DEPOT ([merci Eric Biernat](https://www.datascience.net/fr/challenge/10/details)). Du coup, je fais 3 fold sur les deux dataset et je moyenne les 6 différents scores.

Cette idée étant réellement longue, j'ai installé tout mon environnement sur une instance AWS (AMI publique sur us-east : Environment ML / ami-d25b79b8). J'utilise TMUX pour lancer une session asynchrone et l'API Twilio pour me prévenir par sms que le traitement est terminé.

## Critiques de mon travail
La première critique sur ma démarche est que c'est ici tout ce que j'ai testé pour le challenge.
J'aurais du chercher à tester d'autres modèles, tester pour chaque étape du preprocessing si elle apporte bien une amélioration du score (avec le modèle quick & dirty) et faire un vrai travail de feature engineering (mon preprocessing est un peu léger et reste très classique).

Enfin, l'idée de séparer les données en deux sur la colonne VOIE_DEPOT est, à mon avis, pertinente mais ce qui aurait été vraiment pertinent c'est d'apprendre deux algorithmes différents pour chaque partie.
