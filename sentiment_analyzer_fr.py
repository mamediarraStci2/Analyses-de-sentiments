import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse de Sentiments Twitter",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🐦"
)

# Ajout du logo Twitter en haut de la page
st.image("twitter_logo.png", width=100)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
    }
    .neutral {
        background-color: #e2e3e5;
        color: #383d41;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertForSequenceClassification.from_pretrained("./sentiment_model_final")
    model.eval()
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.detach().numpy()[0]

def main():
    st.title("🐦 Analyse de Sentiments Twitter")
    st.write("Entrez votre texte ci-dessous pour analyser son sentiment.")

    # Interface utilisateur
    text_input = st.text_area("Votre texte :", height=100)
    
    if st.button("Analyser"):
        if text_input.strip():
            with st.spinner('Analyse en cours...'):
                # Chargement du modèle
                tokenizer, model = load_model()
                
                # Analyse
                probabilities = analyze_sentiment(text_input, tokenizer, model)
                sentiment_labels = ["Négatif", "Neutre", "Positif"]
                predicted_class = np.argmax(probabilities)
                
                # Affichage des résultats
                sentiment = sentiment_labels[predicted_class]
                
                # Couleur en fonction du sentiment
                if sentiment == "Positif":
                    box_class = "positive"
                elif sentiment == "Négatif":
                    box_class = "negative"
                else:
                    box_class = "neutral"
                
                st.markdown(f"""
                    <div class='result-box {box_class}'>
                        <h3>Sentiment détecté : {sentiment}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Affichage des probabilités
                st.write("### Probabilités :")
                for label, prob in zip(sentiment_labels, probabilities):
                    st.progress(float(prob))
                    st.write(f"{label}: {prob*100:.2f}%")
        else:
            st.error("Veuillez entrer un texte à analyser.")

if __name__ == "__main__":
    main()
