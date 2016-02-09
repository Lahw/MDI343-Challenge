# MDI343-Challenge
Challenge du cours MDI343 du Mastère Big Data de Telecom.

***Objectif :*** Prédire l'issue d'une demande de brevet déposée auprès de l'Organisme Européen des Brevets (OEB) : délivrance, ou non-délivrance (par rejet de la demande ou abandon du déposant).

## Prérequis
```
pip install -r requirements.txt
```

## Organisation
Ou comment j'ai travaillé ce challenge
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

Tout le preprocessing est présent dans le fichier preprocessing.py. Ma logique a été la suivante :

