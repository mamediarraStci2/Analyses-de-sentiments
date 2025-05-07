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
        model = CamembertForSequenceClassification.from_pretrained('./sentiment_model_final')
        tokenizer = CamembertTokenizer.from_pretrained('./sentiment_model_final')
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

# Interface principale
model, tokenizer = load_model()
metrics = load_metrics()

if model is not None and tokenizer is not None:
    # Création de deux colonnes pour l'interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Interface principale
        st.header("Analyse de sentiment")
        
        # Exemples pré-définis
        st.subheader("Exemples de phrases")
        example_type = st.selectbox("Choisir une catégorie:", ["Positif", "Neutre", "Négatif"])
        selected_example = st.selectbox("Sélectionner un exemple:", example_phrases[example_type])
        
        # Initialiser la session state pour stocker le texte
        if 'text_input' not in st.session_state:
            st.session_state.text_input = ""
        
        # Zone de texte pour l'entrée utilisateur
        user_input = st.text_area("Entrez votre texte en français (ou utilisez un exemple):", 
                                value=st.session_state.text_input,
                                height=100)
        
        # Bouton pour utiliser l'exemple
        if st.button("Utiliser cet exemple"):
            st.session_state.text_input = selected_example
            st.experimental_rerun()
        
        if st.button("Analyser"):
            if user_input:
                try:
                    # Prétraiter le texte
                    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                    
                    # Faire la prédiction
                    with torch.no_grad():
                        outputs = model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        sentiment_idx = predictions.argmax().item()
                        
                    # Afficher les résultats
                    sentiment_map = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                    confidence = predictions[0][sentiment_idx].item()
                    
                    # Couleurs pour les résultats
                    sentiment_colors = {
                        "Positif": "green",
                        "Neutre": "blue",
                        "Négatif": "red"
                    }
                    sentiment_result = sentiment_map[sentiment_idx]
                    
                    st.subheader("Résultats")
                    st.markdown(f"<h3 style='color: {sentiment_colors[sentiment_result]}'>Sentiment: {sentiment_result}</h3>", unsafe_allow_html=True)
                    st.write(f"Confiance: {confidence:.2%}")
                    
                    # Afficher les probabilités pour chaque classe
                    st.subheader("Probabilités par classe")
                    
                    # Créer un graphique de barres pour les probabilités
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sentiments = ["Négatif", "Neutre", "Positif"]
                    colors = ["red", "blue", "green"]
                    probs = [float(predictions[0][i]) for i in range(3)]
                    
                    bars = ax.bar(sentiments, probs, color=colors)
                    ax.set_ylabel('Probabilité')
                    ax.set_title('Probabilités par classe de sentiment')
                    
                    # Ajouter les valeurs au-dessus des barres
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.2%}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
                    # Explication des résultats
                    st.subheader("Interprétation")
                    if max(probs) < 0.96:  # Changé de 0.5 à 0.96 pour correspondre à votre demande
                        st.warning("⚠️ La confiance est insuffisante. Le modèle n'est pas assez certain de ce résultat.")
                        st.markdown("""
                        **Pourquoi le modèle manque-t-il de confiance?**
                        - Texte trop court ou ambigu
                        - Contenu dans un registre de langue non reconnu
                        - Expressions idiomatiques ou sarcastiques difficiles à interpréter
                        - Le modèle nécessite un réentraînement avec plus de données
                        """)
                    else:
                        st.success(f"✅ Le modèle a identifié un sentiment {sentiment_result.lower()} avec une excellente confiance.")
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {str(e)}")
                    st.info("Veuillez réessayer avec un texte plus court ou différent.")
            else:
                st.warning("Veuillez entrer un texte à analyser.")
    
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
else:
    st.error("Le modèle n'a pas pu être chargé. Veuillez vérifier que le modèle est bien entraîné et que les fichiers nécessaires sont présents.") 