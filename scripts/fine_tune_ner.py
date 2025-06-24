"""
Fine-Tune NER Model for Amharic Telegram Data

This script loads CoNLL-formatted data, tokenizes and aligns labels, fine-tunes a pre-trained model, evaluates, and saves the model.

Usage:
    python scripts/fine_tune_ner.py
"""

import os
import pandas as pd
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import numpy as np

# 1. Data Loading & Parsing

def read_conll(filepath: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Read CoNLL-formatted file and return sentences and labels."""
    sentences, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                splits = line.split()
                if len(splits) == 2:
                    tokens.append(splits[0])
                    tags.append(splits[1])
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels

# 2. Tokenization & Label Alignment

def tokenize_and_align_labels(sentences, labels, tokenizer, label2id, max_length=128):
    """Tokenize sentences and align NER labels with tokens."""
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,
    )
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                # For subwords, use I- tag if B-, else same
                tag = label[word_idx]
                if tag.startswith('B-'):
                    tag = 'I-' + tag[2:]
                label_ids.append(label2id.get(tag, label2id[label[word_idx]]))
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    tokenized_inputs['labels'] = aligned_labels
    return tokenized_inputs

# 3. Main Fine-Tuning Pipeline

def main():
    # --- Config ---
    conll_path = 'data/processed/ner_sample_conll.txt'
    model_checkpoint = 'xlm-roberta-base'  # Or 'Davlan/bert-tiny-amharic', 'Davlan/afro-xlmr-base'
    output_dir = './results_ner_amharic'
    num_train_epochs = 3
    batch_size = 8
    learning_rate = 2e-5

    # --- Load Data ---
    sentences, ner_labels = read_conll(conll_path)
    print(f"Loaded {len(sentences)} sentences.")

    # --- Label Mapping ---
    label_list = sorted(set(l for label_seq in ner_labels for l in label_seq))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # --- Prepare Dataset ---
    dataset = Dataset.from_dict({
        'tokens': sentences,
        'ner_tags': ner_labels
    })
    # Split into train/val (e.g., 90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    def preprocess_batch(batch):
        return tokenize_and_align_labels(batch["tokens"], batch["ner_tags"], tokenizer, label2id)

    tokenized_train = dataset['train'].map(
        preprocess_batch,
        batched=True
    )
    tokenized_val = dataset['test'].map(
        preprocess_batch,
        batched=True
    )

    # --- Model ---
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )

    # --- Metrics ---
    from seqeval.metrics import classification_report, f1_score, accuracy_score
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_preds = [
            [id2label[pred] for pred, lab in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return {
            "accuracy": accuracy_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
            "report": classification_report(true_labels, true_preds)
        }

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # --- Train ---
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main() 