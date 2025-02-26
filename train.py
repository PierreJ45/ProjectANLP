import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
)

import wandb

# Initialize wandb
wandb.init(project="KaggleNLP")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Custom Dataset class
class LanguageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Load and preprocess data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    texts = df["Text"].values

    # Create label mapping
    unique_labels = df["Label"].unique()
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    labels = [label_to_id[label] for label in df["Label"]]
    return texts, labels, label_to_id, id_to_label


# Compute metrics function for Trainer
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    metrics = {"accuracy": accuracy, "f1_weighted": f1}

    wandb.log(metrics)
    return metrics


# Main training function
def train_model():
    # Load data
    texts, labels, label_to_id, id_to_label = load_and_prepare_data(
        "train_submission.csv"
    )

    # Split into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    # Initialize tokenizer and model
    model_name = "xlm-roberta-large"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_to_id)
    ).to(device)

    # Create datasets
    train_dataset = LanguageDataset(train_texts, train_labels, tokenizer)
    val_dataset = LanguageDataset(val_texts, val_labels, tokenizer)

    # Training arguments optimized for RTX 4090
    training_args = TrainingArguments(
        output_dir="./results_2",
        num_train_epochs=100,
        per_device_train_batch_size=64,  # Adjust based on VRAM usage
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,  # Effective batch size: 32
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        fp16=True,  # Mixed precision training for RTX 4090
        dataloader_num_workers=4,
        report_to="wandb",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")

    return trainer, id_to_label


# Inference function
def predict(text, model, tokenizer, id_to_label, max_length=128):
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    return id_to_label[prediction]


if __name__ == "__main__":
    # Train the model
    trainer, id_to_label = train_model()
