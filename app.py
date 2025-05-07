import streamlit as st
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration de l'interface
st.set_page_config(page_title="Analyse de Sentiments", page_icon="😊", layout="wide")
st.title("Analyse de Sentiments avec CamemBERT")

# Ajouter des informations sur le modèle
st.markdown("""
## À propos de cette application
Cette application utilise un modèle CamemBERT, spécialisé pour la langue française, pour analyser le sentiment d'un texte.
Le modèle a été entraîné sur des données augmentées pour améliorer sa précision sur des phrases courtes.
""")

# Exemples de phrases
example_phrases = {
    "Positif": [
        "Je t'aime de tout mon cœur",
        "C'est une excellente nouvelle",
        "Ce film était vraiment incroyable",
        "Merci pour ton aide précieuse"
    ],
    "Neutre": [
        "Je vais au cinéma ce soir",
        "Le train part à 15h",
        "J'habite à Paris depuis 5 ans",
        "Le livre contient 300 pages"
    ],
    "Négatif": [
        "Je déteste cette situation",
        "Ce produit est de très mauvaise qualité",
        "Je suis très déçu par ce service",
        "C'était une journée horrible"
    ]
}

# Charger le modèle et le tokenizer
@st.cache_resource
def load_model():
    try:
        # Charger le modèle local
        model_path = os.path.join(os.path.dirname(__file__), "models")
        model = CamembertForSequenceClassification.from_pretrained(model_path)
        tokenizer = CamembertTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None, None

# Charger les métriques
@st.cache_data
def load_metrics():
    try:
        with open('training_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "train_metrics": {"eval_accuracy": 0, "eval_f1": 0},
            "val_metrics": {"eval_accuracy": 0, "eval_f1": 0}
        }

# Charger le modèle
model, tokenizer = load_model()

if model is None or tokenizer is None:
    st.error("Impossible de charger le modèle. L'application ne peut pas continuer.")
    st.stop()

# Fonction pour analyser le sentiment
def analyze_sentiment(text):
    # Tokenisation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # Convertir en numpy pour faciliter la manipulation
    probs = probabilities.numpy()[0]
    
    return {
        "Négatif": float(probs[0]),
        "Neutre": float(probs[1]),
        "Positif": float(probs[2])
    }

# Interface principale
metrics = load_metrics()

# Création de deux colonnes pour l'interface
col1, col2 = st.columns([3, 1])

with col1:
    # Interface principale
    st.header("Analyse de sentiment")
    
    # Exemples pré-définis
    st.subheader("Exemples de phrases")
    sentiment_type = st.selectbox("Choisissez un type de sentiment", list(example_phrases.keys()))
    selected_example = st.selectbox("Sélectionnez une phrase exemple", example_phrases[sentiment_type])
    
    # Initialiser la session state pour stocker le texte
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # Zone de texte pour l'entrée utilisateur
    user_input = st.text_area("Ou écrivez votre propre texte:", value=selected_example, height=100)
    
    if st.button("Analyser"):
        if user_input.strip():
            # Afficher un spinner pendant l'analyse
            with st.spinner("Analyse en cours..."):
                # Obtenir les probabilités
                results = analyze_sentiment(user_input)
                
                # Afficher les résultats
                st.markdown("### Résultats de l'analyse")
                
                # Créer un graphique à barres
                fig, ax = plt.subplots(figsize=(10, 5))
                sentiments = list(results.keys())
                probabilities = list(results.values())
                
                # Définir les couleurs pour chaque sentiment
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                
                # Créer le graphique à barres
                bars = ax.bar(sentiments, probabilities, color=colors)
                
                # Personnaliser le graphique
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probabilité')
                ax.set_title('Distribution des sentiments')
                
                # Ajouter les valeurs sur les barres
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2%}',
                           ha='center', va='bottom')
                
                # Afficher le graphique
                st.pyplot(fig)
                
                # Afficher le sentiment dominant
                dominant_sentiment = max(results.items(), key=lambda x: x[1])[0]
                st.markdown(f"### Sentiment dominant : **{dominant_sentiment}**")
                
                # Afficher toutes les probabilités
                st.markdown("### Détail des probabilités :")
                for sentiment, probability in results.items():
                    st.write(f"{sentiment}: {probability:.2%}")
        else:
            st.error("Veuillez entrer un texte à analyser.")

with col2:
    # Afficher les métriques si elles existent
    if any(metrics["train_metrics"].values()) or any(metrics["val_metrics"].values()):
        st.sidebar.header("Métriques du modèle")
        st.sidebar.subheader("Métriques d'entraînement")
        st.sidebar.write(f"Accuracy: {metrics['train_metrics']['eval_accuracy']:.2%}")
        st.sidebar.write(f"F1 Score: {metrics['train_metrics']['eval_f1']:.2%}")
        
        st.sidebar.subheader("Métriques de validation")
        st.sidebar.write(f"Accuracy: {metrics['val_metrics']['eval_accuracy']:.2%}")
        st.sidebar.write(f"F1 Score: {metrics['val_metrics']['eval_f1']:.2%}")
        
        # Ajouter des conseils pour l'analyse
        st.sidebar.header("Conseils d'utilisation")
        st.sidebar.markdown("""
        Pour de meilleurs résultats:
        - Utilisez des phrases complètes
        - Évitez le sarcasme ou l'ironie
        - Plus le texte est clair, meilleure sera l'analyse
        """) 