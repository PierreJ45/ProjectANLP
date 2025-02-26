import torch
import pandas as pd

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, ClassLabel

from models import Model
from utils import Language


class DeepModel(Model):
    def __init__(
        self,
        labels: list[str],
        model: str = "xlm-roberta-base",
        tokenizer: str = "xlm-roberta-base",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(labels)
        self.label_encoder = ClassLabel(names=self.labels)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=len(labels)
        ).to(device)

    def infer(self, text: str) -> Language:
        output = self.model(**self.tokenizer(text, return_tensors="pt").to(self.device))
        return self.labels[int(torch.argmax(output.logits).item())]

    def tokenize(self, df: Dataset):
        return self.tokenizer(df["text"], padding="max_length", truncation=True)

    def encode_labels(self, df: Dataset):
        return {"label": self.label_encoder.str2int(df["label"])}

    def train(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        epochs: int = 1,
        batch_size: int = 8,
        lr: float = 5e-5,
    ):
        train_dataset = (
            Dataset.from_dict({"text": train_df["Text"], "label": train_df["Label"]})
            .map(self.tokenize, batched=True)
            .map(self.encode_labels, batched=True)
        )

        validation_dataset = (
            Dataset.from_dict(
                {"text": validation_df["Text"], "label": validation_df["Label"]}
            )
            .map(self.tokenize, batched=True)
            .map(self.encode_labels, batched=True)
        )

        # Set dataset format for PyTorch
        train_dataset.set_format(
            type="torch", columns=["label", "input_ids", "attention_mask"]
        )
        validation_dataset.set_format(
            type="torch", columns=["label", "input_ids", "attention_mask"]
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # type: ignore
        val_loader = DataLoader(
            validation_dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=False,
        )

        for param in self.model.base_model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Optimizer and loss function
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in tqdm(range(epochs)):
            total_loss = 0

            for batch in tqdm(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = self.model(**{k: v for k, v in batch.items() if k != "label"})
                loss = loss_fn(outputs.logits, batch["label"])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

            # Validation
            self.model.eval()
            correct, total = 0, 0
            train_correct, train_total = 0, 0
            val_loss = 0

            with torch.no_grad():
                for batch in train_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(
                        **{k: v for k, v in batch.items() if k != "label"}
                    )

                    predictions = torch.argmax(outputs.logits, dim=-1)
                    train_correct += (predictions == batch["label"]).sum().item()
                    train_total += batch["label"].size(0)

                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(
                        **{k: v for k, v in batch.items() if k != "label"}
                    )
                    loss = loss_fn(outputs.logits, batch["label"])
                    val_loss += loss.item()

                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch["label"]).sum().item()
                    total += batch["label"].size(0)

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            train_acc = train_correct / train_total
            print(
                f"Validation Loss: {avg_val_loss:.4f} - Accuracy: {accuracy:.4%} - Train Accuracy: {train_acc:.4%}"
            )

        self.model.save_pretrained("./fine_tuned_xlm_roberta")
        self.tokenizer.save_pretrained("./fine_tuned_xlm_roberta")
