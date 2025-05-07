import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os
from torch.utils.data import Dataset
import random
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration optimisée pour la mémoire
MODEL_NAME = "camembert-base"
MAX_LENGTH = 128  # Réduit pour économiser la mémoire
BATCH_SIZE = 4    # Réduit pour économiser la mémoire
LEARNING_RATE = 1e-5
NUM_EPOCHS = 15
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.02
EARLY_STOPPING_PATIENCE = 5
GRADIENT_ACCUMULATION_STEPS = 8  # Augmenté pour compenser la réduction du batch size

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

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

def load_and_prepare_data():
    print("Chargement des données...")
    # Charger les données augmentées
    train_df = pd.read_csv('augmented_data/train_augmented.csv')
    val_df = pd.read_csv('augmented_data/val_augmented.csv')

    # Convertir les sentiments en labels numériques
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_df['label'] = train_df['sentiment'].map(sentiment_map)
    val_df['label'] = val_df['sentiment'].map(sentiment_map)

    # Afficher la distribution des classes
    print("\nDistribution des classes d'entraînement:")
    print(train_df['sentiment'].value_counts())
    print("\nDistribution des classes de validation:")
    print(val_df['sentiment'].value_counts())

    return train_df, val_df

def train_model():
    print("Initialisation du modèle...")
    # Charger le tokenizer et le modèle
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
    model = CamembertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # Charger et préparer les données
    train_df, val_df = load_and_prepare_data()

    # Créer les datasets
    train_dataset = SentimentDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    val_dataset = SentimentDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )

    # Configuration de l'entraînement optimisée
    training_args = TrainingArguments(
        output_dir='./sentiment_model_final',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,  # Désactivé car pas de GPU CUDA
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine_with_restarts",
        save_total_limit=3,
        report_to="tensorboard",
        dataloader_num_workers=0,  # Réduit l'utilisation de la mémoire
        dataloader_pin_memory=False,  # Réduit l'utilisation de la mémoire
    )

    # Initialiser le trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    print("Début de l'entraînement...")
    # Entraîner le modèle
    trainer.train()

    # Évaluer le modèle final
    print("Évaluation du modèle final...")
    final_metrics = trainer.evaluate()
    print(f"Métriques finales: {final_metrics}")

    # Sauvegarder les métriques
    metrics = {
        "train_metrics": {
            "eval_accuracy": final_metrics["eval_accuracy"],
            "eval_f1": final_metrics["eval_f1"],
            "eval_precision": final_metrics["eval_precision"],
            "eval_recall": final_metrics["eval_recall"]
        },
        "val_metrics": {
            "eval_accuracy": final_metrics["eval_accuracy"],
            "eval_f1": final_metrics["eval_f1"],
            "eval_precision": final_metrics["eval_precision"],
            "eval_recall": final_metrics["eval_recall"]
        }
    }

    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Sauvegarder le modèle
    trainer.save_model('./sentiment_model_final')
    tokenizer.save_pretrained('./sentiment_model_final')

    print("Entraînement terminé et modèle sauvegardé.")

    # Visualiser les résultats
    plot_training_results(trainer.state.log_history)

    return final_metrics

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
    
    # Graphique de la comparaison de métriques finales sous forme de barres
    if eval_accuracy and eval_f1:
        plt.figure(figsize=(10, 6))
        final_metrics = {
            'Accuracy': eval_accuracy[-1],
            'F1 Score': eval_f1[-1]
        }
        sns.barplot(x=list(final_metrics.keys()), y=list(final_metrics.values()))
        plt.ylim(0, 1)
        plt.title('Final Model Performance')
        plt.savefig('visualizations/final_metrics.png')

if __name__ == "__main__":
    train_model() 