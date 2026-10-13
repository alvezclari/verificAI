"""Inferência local de um checkpoint HuggingFace já ajustado para a tarefa."""

import json
from pathlib import Path
import threading

import numpy as np

from backend.ml.ensemble import softmax, validate_probabilities


class TransformerClassifier:
    def __init__(self, checkpoint, device="cpu", batch_size=8):
        checkpoint = Path(checkpoint)
        metadata = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1 or metadata.get("labels") != {"0": "fake", "1": "confiavel"}:
            raise ValueError("Checkpoint sem mapeamento de classes validado pelo treinamento.")
        self.max_length = int(metadata["max_length"])
        self.metadata = metadata
        self.temperature = float(metadata.get("temperature", 1.0))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperatura de calibração inválida.")
        if self.max_length <= 0 or batch_size <= 0:
            raise ValueError("max_length e batch_size devem ser positivos.")
        # Dependências pesadas só são importadas ao ativar o ensemble.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self._lock = threading.RLock()
        self.batch_size = batch_size
        model_config = json.loads((checkpoint / "config.json").read_text())
        # Preserva a tokenização SentencePiece do checkpoint DeBERTa.
        tokenizer_options = {"fix_mistral_regex": False} if model_config.get("model_type") == "deberta-v2" else {}
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, **tokenizer_options)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
        if self.model.config.num_labels != 2 or self.model.config.id2label != {0: "fake", 1: "confiavel"}:
            raise ValueError("Cabeça do Transformer incompatível com o ensemble.")
        if self.max_length > getattr(self.model.config, "max_position_embeddings", self.max_length):
            raise ValueError("max_length excede a capacidade do Transformer.")
        self.model.to(self.device)
        self.model.eval()

    def predict_logits(self, texts):
        texts = list(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Informe textos não vazios.")
        outputs = []
        with self._lock, self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                inputs = self.tokenizer(
                    texts[start:start + self.batch_size], padding=True,
                    truncation=True, max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits.detach().float().cpu().numpy()
                outputs.append(logits)
        return np.concatenate(outputs)

    def predict_proba(self, texts):
        return validate_probabilities(softmax(self.predict_logits(texts) / self.temperature))

    def input_info(self, text):
        with self._lock:
            tokens = self.tokenizer(text, truncation=False, return_attention_mask=False, verbose=False)["input_ids"]
        return {"input_tokens": len(tokens), "used_tokens": min(len(tokens), self.max_length),
                "truncated": len(tokens) > self.max_length}
