import pandas as pd
import numpy as np
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', 
                                 max_length=max_length, return_tensors="pt")
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def translate_english_to_french(texts):
    """
    Cette fonction devrait être remplacée par une méthode de traduction plus avancée
    si vous avez un service de traduction disponible. Pour l'instant, gardons-la simple.
    """
    # Exemple très basique - à remplacer par une vraie API de traduction
    # Ici on suppose que les données sont déjà en français ou qu'on les utilise telles quelles
    return texts

def train_model():
    print("Chargement des données...")
    # Charger les données
    train_df = pd.read_csv('tweet-sentiment-extraction/train.csv')
    test_df = pd.read_csv('tweet-sentiment-extraction/test.csv')

    # Filtrer les données invalides et traiter les valeurs manquantes
    train_df = train_df.dropna(subset=['text', 'sentiment'])
    
    # Équilibrer les classes si nécessaire
    class_counts = train_df['sentiment'].value_counts()
    print(f"Distribution des classes avant équilibrage: {class_counts}")
    
    # Augmenter les données pour les classes minoritaires
    min_class_count = class_counts.min()
    balanced_dfs = []
    
    for sentiment, count in class_counts.items():
        class_df = train_df[train_df['sentiment'] == sentiment]
        if count < class_counts.max():
            # Suréchantillonnage simple - peut être amélioré avec des techniques plus avancées
            class_df = class_df.sample(class_counts.max(), replace=True, random_state=42)
        balanced_dfs.append(class_df)
    
    train_df = pd.concat(balanced_dfs)
    print(f"Distribution des classes après équilibrage: {train_df['sentiment'].value_counts()}")

    # Traduire en français si nécessaire (ou utiliser des données françaises)
    texts = translate_english_to_french(train_df['text'].values)
    labels = train_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values

    # Diviser en train, validation et test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"Taille des ensembles - Train: {len(train_texts)}, Validation: {len(val_texts)}, Test: {len(test_texts)}")

    # Initialiser le tokenizer et le modèle
    print("Initialisation du modèle CamemBERT...")
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=3)

    # Créer les datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    # Définir les arguments d'entraînement
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),  # Utiliser la précision mixte si disponible
        gradient_accumulation_steps=2,  # Pour simuler de plus grands batch sizes
        eval_accumulation_steps=4
    )

    # Initialiser le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Entraîner le modèle
    print("Début de l'entraînement...")
    trainer.train()

    # Évaluer le modèle sur l'ensemble de test
    print("Évaluation du modèle...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Résultats sur l'ensemble de test: {test_results}")

    # Sauvegarder le modèle final
    print("Sauvegarde du modèle...")
    model.save_pretrained('./sentiment_model_final')
    tokenizer.save_pretrained('./sentiment_model_final')

    # Sauvegarder les métriques
    metrics = {
        'train_metrics': {'eval_accuracy': trainer.state.log_history[-2]['eval_accuracy'], 
                          'eval_f1': trainer.state.log_history[-2]['eval_f1']},
        'val_metrics': {'eval_accuracy': test_results['eval_accuracy'], 
                        'eval_f1': test_results['eval_f1']}
    }

    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print("Entraînement terminé!")
    
    # Visualiser les résultats
    plot_training_results(trainer.state.log_history)

def plot_training_results(log_history):
    # Extraire les métriques d'entraînement
    train_loss = [x.get('loss') for x in log_history if x.get('loss') is not None]
    eval_loss = [x.get('eval_loss') for x in log_history if x.get('eval_loss') is not None]
    eval_accuracy = [x.get('eval_accuracy') for x in log_history if x.get('eval_accuracy') is not None]
    eval_f1 = [x.get('eval_f1') for x in log_history if x.get('eval_f1') is not None]
    
    # Créer un dossier pour les visualisations
    os.makedirs('visualizations', exist_ok=True)
    
    # Graphique des pertes
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_loss)), train_loss, label='Train Loss')
    plt.plot(range(len(eval_loss)), eval_loss, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('visualizations/loss_curve.png')
    
    # Graphique des métriques
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(eval_accuracy)), eval_accuracy, label='Accuracy')
    plt.plot(range(len(eval_f1)), eval_f1, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.savefig('visualizations/metrics_curve.png')

if __name__ == "__main__":
    train_model() 