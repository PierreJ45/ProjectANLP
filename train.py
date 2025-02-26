import torch
import wandb

from transformers import (
    Trainer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
)

from data import get_tokenized_datasets, LanguageDataset
from eval import compute_metrics


# Initialize wandb
wandb.init(project="KaggleNLP")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Main training function
def train_model(
    model_name: str,
    train_dataset: LanguageDataset,
    val_dataset: LanguageDataset,
    n_labels: int,
):
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels
    ).to(device)

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

    return trainer


# Inference function
def predict(
    text: str,
    model: torch.nn.Module,
    tokenizer: XLMRobertaTokenizer,
    idx_to_label: dict[int, str],
    max_length=128,
):
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
        prediction = int(torch.argmax(logits, dim=-1).item())

    return idx_to_label[prediction]


if __name__ == "__main__":
    model_name = "xlm-roberta-large"

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset, labels, idx_to_label = get_tokenized_datasets(tokenizer)

    trainer = train_model(model_name, train_dataset, val_dataset, len(labels))
