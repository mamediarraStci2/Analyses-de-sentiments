import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(preds, labels):
    preds = preds.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    # Charger les données
    train_df = pd.read_csv('tweet-sentiment-extraction/train.csv')
    test_df = pd.read_csv('tweet-sentiment-extraction/test.csv')

    # Préparer les données
    texts = train_df['text'].values
    labels = train_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values

    # Diviser en train et validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Initialiser le tokenizer et le modèle
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # Créer les datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Configuration de l'entraînement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3

    # Entraînement
    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})

        # Évaluation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = outputs.logits
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        # Calculer les métriques
        metrics = compute_metrics(np.array(val_preds), np.array(val_labels))
        print(f"Epoch {epoch + 1} - Validation metrics:", metrics)
        
        # Sauvegarder le meilleur modèle
        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            model.save_pretrained('./sentiment_model_bert')
            tokenizer.save_pretrained('./sentiment_model_bert')

    # Évaluer sur l'ensemble de test
    test_texts = test_df['text'].values
    test_dataset = SentimentDataset(test_texts, [0] * len(test_texts), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits
            
            test_preds.extend(preds.cpu().numpy())

    # Sauvegarder les métriques
    metrics = {
        'train_metrics': compute_metrics(np.array(val_preds), np.array(val_labels)),
        'val_metrics': metrics,
        'test_metrics': {
            'predictions': np.array(test_preds).argmax(-1).tolist()
        }
    }

    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    train_model() 