"""Leitura de texto cru e splits reprodutíveis compartilhados pelos dois modelos."""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split

LABELS = {"0": "fake", "1": "confiavel"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_records(path, text_column, label_column, real_label, zip_member=None):
    """CSV UTF-8; real_label é obrigatório para evitar inferir a semântica de 0/1."""
    path = Path(path)
    if zip_member:
        with zipfile.ZipFile(path) as archive:
            content = archive.read(zip_member)
    else:
        content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if not {text_column, label_column}.issubset(reader.fieldnames or []):
        raise ValueError(f"Colunas ausentes: {text_column}, {label_column}.")
    rows = list(reader)
    raw_labels = {row[label_column].strip() for row in rows}
    if len(raw_labels) != 2 or str(real_label) not in raw_labels:
        raise ValueError("Dataset deve ter duas classes e incluir --real-label.")
    texts, labels, seen = [], [], {}
    for row in rows:
        text = row[text_column].strip()
        if not text:
            raise ValueError("Dataset contém texto vazio.")
        label = int(row[label_column].strip() == str(real_label))
        key = fingerprint(" ".join(text.casefold().split()))
        if key in seen:
            if seen[key] != label:
                raise ValueError("Texto duplicado com rótulos conflitantes.")
            continue
        seen[key] = label
        texts.append(text)
        labels.append(label)
    return texts, labels, {
        "source": str(path.resolve()), "zip_member": zip_member,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "text_column": text_column, "label_column": label_column,
        "real_label": str(real_label), "duplicates_removed": len(rows) - len(texts),
    }


def split_records(texts, labels, seed=42, validation_size=0.2, test_size=0.2):
    if not 0 < validation_size < 1 or not 0 < test_size < 1 or validation_size + test_size >= 1:
        raise ValueError("Frações de validação/teste inválidas.")
    counts = Counter(labels)
    if set(counts) != {0, 1} or min(counts.values()) < 3:
        raise ValueError("Cada classe deve ter ao menos três textos distintos.")
    indices = np.arange(len(texts))
    labels = np.asarray(labels, dtype=int)
    train_val, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
    train, val = train_test_split(
        train_val, test_size=validation_size / (1 - test_size),
        random_state=seed, stratify=labels[train_val],
    )
    splits = {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}
    for name, group in splits.items():
        if set(labels[group]) != {0, 1}:
            raise ValueError(f"Split {name} deve conter ambas as classes; use mais dados.")
    metadata = {
        "schema_version": 1, "labels": LABELS,
        "dataset_sha256": fingerprint(list(zip(texts, labels.tolist()))),
        "seed": seed, "validation_size": validation_size, "test_size": test_size,
        "indices": splits,
        "record_hashes": [fingerprint([text, int(label)]) for text, label in zip(texts, labels)],
    }
    metadata["split_signature"] = fingerprint(metadata)
    return splits, metadata
