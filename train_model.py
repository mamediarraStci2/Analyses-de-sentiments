from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_sample_dataset():
    # Création d'un petit dataset d'exemple en français
    data = {
        'text': [
            "J'adore ce film, c'est vraiment magnifique !",
            "Ce restaurant est excellent, je recommande vivement.",
            "Je déteste cette expérience, c'était horrible.",
            "Le service client est déplorable.",
            "C'était correct, rien de spécial.",
            "Je ne sais pas trop quoi en penser.",
            "Super expérience, à refaire !",
            "Très déçu par la qualité.",
            "Moyen, ça pourrait être mieux.",
            "Extraordinaire, je suis conquis !"
        ],
        'label': [2, 2, 0, 0, 1, 1, 2, 0, 1, 2]  # 0: négatif, 1: neutre, 2: positif
    }
    return pd.DataFrame(data)

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    correct = sum(1 for p, a in zip(predictions, actual_labels) if p == a)
    total = len(predictions)
    accuracy = (correct / total) * 100
    return accuracy

def train():
    # Création du dataset
    df = create_sample_dataset()
    
    # Split en train/validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Taille du dataset d'entraînement: {len(train_df)} exemples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Taille du dataset de test: {len(val_df)} exemples ({len(val_df)/len(df)*100:.1f}%)")
    
    # Chargement du tokenizer et du modèle
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=3)
    
    # Création des datasets
    train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
    
    # Création des dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Optimiseur
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Entraînement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        
        # Évaluation à la fin de chaque époque
        train_accuracy = evaluate_model(model, train_loader, device)
        val_accuracy = evaluate_model(model, val_loader, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  Accuracy sur entraînement: {train_accuracy:.1f}%')
        print(f'  Accuracy sur test: {val_accuracy:.1f}%')
    
    # Sauvegarde du modèle
    model.save_pretrained("./sentiment_model_final")
    tokenizer.save_pretrained("./sentiment_model_final")
    print("\nModèle sauvegardé avec succès !")

if __name__ == "__main__":
    train()
