# Concevoir un classifieur de détection automatique de SPAM.

La collection SMS Spam est un ensemble de messages SMS marqués qui ont été collectés pour la recherche sur les SMS Spam. Elle contient un ensemble de messages SMS en anglais de 5 574 messages, étiquetés selon qu'ils sont ham (légitimes) ou spam.
Je vous encourage à vous documenter sur les caractéristiques type des spam et de développer votre stratégie de préparation des données dans ce sens.

En tant que développeur IA, voici les missions :
- Analyse du besoin
- Construction d'un pipeline de ML
- Prétraitement des données
- Entrainement, fine tuning, validation et sélection d'un modèle de classification

# Contexte du projet
Les fichiers contiennent un message par ligne. Chaque ligne est composée de deux colonnes : v1 contient le label (ham ou spam) et v2 contient le texte brut.

Ce corpus a été collecté à partir de sources libres ou gratuites pour la recherche sur Internet : https://archive.ics.uci.edu/ml/datasets/sms+spam+collection# https://www.kaggle.com/uciml/sms-spam-collection-dataset

# Livrables
* créer un/des notebook reproductible, commenté, expliqué (IMPORTANT !)
* créer un repo git et un espace sur github/gitlab pour le projet (code refactorisé)
* faire une présentation (slides) qui explique votre démarche et les résultats obtenus avec :
- un document technique qui explique l'outil
- la procédure suivie pour préparer les données et le preprocessing
- la procédure suivie pour trouver un modèle adapté
- le modèle d'IA sélectionné

## BONUS :
Application streamlit qui fait de la prédiction en temps réel d'un message déposé par l'utilisateur

# Critères de performance :
- compréhension du jeux de données
- capacité à préparer les données
- performance des modèles de prédiction
- capacité à apporter une solution dans le temps imparti
- rédaction du notebook
- qualité du synthèse du travail
