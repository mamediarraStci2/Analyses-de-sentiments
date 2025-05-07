# Analyse de Sentiments en Français

Ce projet implémente un système d'analyse de sentiments en français utilisant le modèle CamemBERT. Il permet d'analyser le sentiment (positif, négatif ou neutre) d'un texte en français.

## Fonctionnalités

- Analyse de sentiments en français
- Interface utilisateur avec Streamlit
- Modèle basé sur CamemBERT
- Augmentation de données pour améliorer les performances
- Visualisation des résultats

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/mamediarraStci2/Analyses-de-sentiments.git
cd Analyses-de-sentiments
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancer l'application Streamlit :
```bash
streamlit run app.py
```

2. Entrer un texte en français dans l'interface
3. Le modèle analysera le sentiment et affichera le résultat

## Structure du Projet

- `app.py` : Application Streamlit pour l'interface utilisateur
- `train_camembert_with_augmented.py` : Script d'entraînement du modèle avec données augmentées
- `data_augmentation.py` : Script pour l'augmentation des données
- `requirements.txt` : Liste des dépendances Python

## Modèle

Le modèle utilise CamemBERT, une version française de BERT, fine-tunée pour la classification de sentiments. Il est entraîné sur un ensemble de données en français et utilise des techniques d'augmentation de données pour améliorer ses performances.

## Licence

Ce projet est sous licence MIT. 