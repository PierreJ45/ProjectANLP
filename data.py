import pandas as pd
import torch

# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class LanguageDataset(Dataset):
    def __init__(
        self,
        texts,
        label_idxs,
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.label_idxs = label_idxs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = self.label_idxs[idx]

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
def get_train_data(val_size: float = 0.1, random_state: int = 42):
    """
    Returns train_texts, val_texts, train_label_idxs, val_label_idxs, labels, idx_to_label
    """
    df = pd.read_csv("train_submission.csv")

    # Create label mapping
    labels: list[str] = df["Label"].unique().tolist()
    label_to_idx: dict[str, int] = {label: idx for idx, label in enumerate(labels)}
    idx_to_label: dict[int, str] = {idx: label for label, idx in label_to_idx.items()}

    df["Idx"] = df["Label"].map(label_to_idx)

    val_df = df.sample(frac=val_size, random_state=random_state)
    df = df.drop(val_df.index)

    return df, val_df, labels, idx_to_label


def get_tokenized_datasets(tokenizer, val_size: float = 0.1, random_state: int = 42):
    train_df, val_df, labels, idx_to_label = get_train_data(val_size, random_state)

    train_dataset = LanguageDataset(train_df["Text"].values, train_df["Idx"], tokenizer)
    val_dataset = LanguageDataset(val_df["Text"].values, val_df["Idx"], tokenizer)

    return train_dataset, val_dataset, labels, idx_to_label


def get_test_data():
    """
    Returns a dataframe
    """
    test_df = pd.read_csv("test_without_labels.csv")

    return test_df
