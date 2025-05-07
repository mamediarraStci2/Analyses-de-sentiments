import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import random
import json
import os

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Téléchargement spécifique pour le français
nltk.download('punkt')  # Assurons-nous que punkt est bien installé
nltk.download('stopwords')

def clean_text(text):
    """Nettoyer le texte"""
    if isinstance(text, str):
        text = re.sub(r'@\w+', '', text)  # Supprimer les mentions
        text = re.sub(r'http\S+', '', text)  # Supprimer les URLs
        text = re.sub(r'#', '', text)  # Supprimer les hashtags
        text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
        text = text.lower()  # Mettre en minuscules
        return text.strip()
    return ""

def synonym_replacement(words, n=1):
    """Remplacer n mots aléatoires par des synonymes"""
    # Cette fonction devrait être améliorée avec un vrai dictionnaire de synonymes français
    # Voici un exemple simplifié
    french_synonyms = {
        'bon': ['excellent', 'super', 'formidable', 'agréable'],
        'mauvais': ['terrible', 'horrible', 'affreux', 'médiocre'],
        'aimer': ['adorer', 'apprécier', 'chérir'],
        'détester': ['haïr', 'abhorrer', 'exécrer'],
        'content': ['heureux', 'joyeux', 'satisfait', 'ravi'],
        'triste': ['malheureux', 'mélancolique', 'déprimé', 'abattu'],
        'beau': ['joli', 'magnifique', 'superbe', 'splendide'],
        'laid': ['moche', 'disgracieux', 'hideux', 'repoussant']
    }
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word in french_synonyms]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = french_synonyms[random_word]
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return new_words

def random_insertion(words, n=1):
    """Insérer n mots aléatoires"""
    # Liste basique de mots français courants
    common_words = ['très', 'vraiment', 'plutôt', 'assez', 'tellement', 'complètement', 
                   'absolument', 'totalement', 'franchement', 'sincèrement']
    
    new_words = words.copy()
    for _ in range(n):
        add_word = random.choice(common_words)
        position = random.randint(0, len(new_words))
        new_words.insert(position, add_word)
        
    return new_words

def random_swap(words, n=1):
    """Échanger la position de n paires de mots"""
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) >= 2:
            pos1 = random.randint(0, len(new_words)-1)
            pos2 = random.randint(0, len(new_words)-1)
            new_words[pos1], new_words[pos2] = new_words[pos2], new_words[pos1]
    return new_words

def random_deletion(words, p=0.1):
    """Supprimer des mots aléatoirement avec une probabilité p"""
    # Si la phrase n'a qu'un seul mot, le conserver
    if len(words) == 1:
        return words
    
    # Filtrer les mots à conserver
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    
    # Si tous les mots ont été supprimés, en conserver un au hasard
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]
    
    return new_words

def augment_text(text, sentiment, num_aug=4):
    """Augmenter un texte avec plusieurs techniques"""
    # Utiliser un simple split pour la tokenization, pas besoin de ressources spécifiques
    words = text.split()
    if len(words) <= 3:  # Ne pas augmenter les textes trop courts
        return []
    
    augmented_texts = []
    
    # Appliquer différentes techniques d'augmentation
    for _ in range(num_aug):
        technique = random.choice([1, 2, 3, 4])
        if technique == 1:  # Remplacement de synonymes
            a_words = synonym_replacement(words, n=max(1, int(len(words)*0.1)))
        elif technique == 2:  # Insertion aléatoire
            a_words = random_insertion(words, n=max(1, int(len(words)*0.1)))
        elif technique == 3:  # Échange aléatoire
            a_words = random_swap(words, n=max(1, int(len(words)*0.1)))
        else:  # Suppression aléatoire
            a_words = random_deletion(words, p=0.1)
            
        augmented_text = ' '.join(a_words)
        augmented_texts.append((augmented_text, sentiment))
    
    return augmented_texts

def create_custom_french_data():
    """Créer des données personnalisées en français"""
    custom_data = [
        # Exemples positifs
        ("J'ai passé une journée extraordinaire, tout était parfait!", "positive"),
        ("Ce film est incroyable, je le recommande vivement.", "positive"),
        ("Je suis très heureux aujourd'hui, tout se passe bien.", "positive"),
        ("Le restaurant était excellent, la nourriture délicieuse.", "positive"),
        ("Cette expérience a dépassé toutes mes attentes.", "positive"),
        ("Je t'aime de tout mon cœur.", "positive"),
        ("Tu es la meilleure personne que je connaisse.", "positive"),
        ("Merci beaucoup pour ton aide précieuse.", "positive"),
        ("C'est une nouvelle fantastique, je suis ravi!", "positive"),
        ("Je suis fier de ce que nous avons accompli ensemble.", "positive"),
        
        # Exemples neutres
        ("Je vais faire des courses ce weekend.", "neutral"),
        ("Le train part à 15h demain.", "neutral"),
        ("J'ai regardé un documentaire hier soir.", "neutral"),
        ("Il fait 20 degrés aujourd'hui.", "neutral"),
        ("Le magasin est ouvert jusqu'à 19h.", "neutral"),
        ("Je vais lire ce livre pendant les vacances.", "neutral"),
        ("La réunion est prévue pour lundi prochain.", "neutral"),
        ("J'habite à Paris depuis cinq ans.", "neutral"),
        ("Le film dure environ deux heures.", "neutral"),
        ("Je prends le métro tous les jours.", "neutral"),
        
        # Exemples négatifs
        ("Cette journée a été horrible, rien ne s'est passé comme prévu.", "negative"),
        ("Je suis vraiment déçu par ce service client déplorable.", "negative"),
        ("Je déteste quand les gens sont impolis et irrespectueux.", "negative"),
        ("Ce produit est de très mauvaise qualité, à éviter absolument.", "negative"),
        ("Je me sens triste et déprimé aujourd'hui.", "negative"),
        ("Je n'ai pas aimé ce film, c'était ennuyeux et prévisible.", "negative"),
        ("Cette situation m'angoisse terriblement.", "negative"),
        ("Je suis furieux contre cette décision injuste.", "negative"),
        ("Cette nouvelle m'a brisé le cœur.", "negative"),
        ("Je regrette sincèrement cette erreur.", "negative")
    ]
    
    # Créer un DataFrame
    custom_df = pd.DataFrame(custom_data, columns=['text', 'sentiment'])
    return custom_df

def augment_dataset():
    """Augmenter le jeu de données complet"""
    print("Chargement des données originales...")
    try:
        train_df = pd.read_csv('tweet-sentiment-extraction/train.csv')
        # Nettoyage des données
        train_df = train_df.dropna(subset=['text', 'sentiment'])
        print(f"Données originales chargées: {len(train_df)} exemples")
    except Exception as e:
        print(f"Erreur lors du chargement des données originales: {e}")
        train_df = pd.DataFrame(columns=['text', 'sentiment'])
        print("Création d'un DataFrame vide")
    
    # Créer des données personnalisées en français
    print("Création de données personnalisées en français...")
    custom_df = create_custom_french_data()
    print(f"Données personnalisées créées: {len(custom_df)} exemples")
    
    # Appliquer le nettoyage de texte aux données personnalisées
    custom_df['clean_text'] = custom_df['text'].apply(clean_text)
    
    # Combiner les données existantes avec les données personnalisées
    combined_df = pd.concat([train_df, custom_df], ignore_index=True)
    
    # Nettoyer les textes des données originales
    combined_df.loc[~combined_df.index.isin(custom_df.index), 'clean_text'] = combined_df.loc[~combined_df.index.isin(custom_df.index), 'text'].apply(clean_text)
    
    # Augmenter les données
    print("Augmentation des données...")
    augmented_data = []
    
    # Augmenter principalement les données personnalisées (qui sont en français)
    for idx, row in custom_df.iterrows():
        # Augmenter davantage les exemples courts pour avoir plus de variété
        num_aug = 5 if len(row['text'].split()) < 10 else 3
        augmented_texts = augment_text(row['clean_text'], row['sentiment'], num_aug=num_aug)
        augmented_data.extend(augmented_texts)
    
    # Créer un DataFrame avec les données augmentées
    augmented_df = pd.DataFrame(augmented_data, columns=['text', 'sentiment'])
    
    # Combiner avec les données originales
    final_df = pd.concat([combined_df[['text', 'sentiment']], augmented_df], ignore_index=True)
    
    # Équilibrer les classes
    class_counts = final_df['sentiment'].value_counts()
    print(f"Distribution des classes: {class_counts}")
    
    # Sauvegarder les données augmentées
    os.makedirs('augmented_data', exist_ok=True)
    final_df.to_csv('augmented_data/train_augmented.csv', index=False)
    print(f"Données augmentées sauvegardées: {len(final_df)} exemples")
    
    # Créer un sous-ensemble de validation
    val_size = min(int(len(final_df) * 0.2), 1000)  # Maximum 1000 exemples pour la validation
    val_df = final_df.sample(val_size, random_state=42)
    val_df.to_csv('augmented_data/val_augmented.csv', index=False)
    print(f"Données de validation sauvegardées: {len(val_df)} exemples")
    
    return final_df

if __name__ == "__main__":
    augment_dataset() 