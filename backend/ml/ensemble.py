"""Soft voting de probabilidades; margens não calibradas são recusadas."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import joblib
import numpy as np


def validate_probabilities(values, size=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Esperadas probabilidades de duas classes [fake, confiável].")
    if size is not None and len(values) != size:
        raise ValueError("Quantidade de predições diferente da quantidade de textos.")
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values > 1).any() or not np.allclose(values.sum(axis=1), 1)):
        raise ValueError("Probabilidades inválidas.")
    return values


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 2 or not np.isfinite(logits).all():
        raise ValueError("Esperados dois logits finitos por texto.")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def validate_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (2,) or not np.isfinite(weights).all()
            or (weights <= 0).any() or not np.isfinite(weights.sum())):
        raise ValueError("Informe dois pesos positivos e finitos.")
    return weights / weights.sum()


class LexicalClassifier:
    def __init__(self, model, reliable_class=1, vectorizer=None):
        if not hasattr(model, "predict_proba"):
            raise ValueError("Soft voting exige predict_proba; calibre SVC antes de utilizá-lo.")
        classes = list(model.classes_)
        if len(classes) != 2 or reliable_class not in classes:
            raise ValueError("Mapeamento binário de classes léxicas inválido.")
        self.model = model
        self.vectorizer = vectorizer
        reliable_index = classes.index(reliable_class)
        self.order = [1 - reliable_index, reliable_index]

    def predict_proba(self, texts):
        inputs = self.vectorizer.transform(texts) if self.vectorizer is not None else texts
        return validate_probabilities(self.model.predict_proba(inputs)[:, self.order], len(texts))


class SoftVotingEnsemble:
    def __init__(self, lexical, semantic, weights=(0.5, 0.5)):
        self.lexical = lexical
        self.semantic = semantic
        self.weights = validate_weights(weights)

    @classmethod
    def from_manifest(cls, path, device="cpu"):
        from backend.ml.transformer import TransformerClassifier

        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") == 2:
            from backend.ml.deep_ensemble import CalibratedDeepEnsemble
            return CalibratedDeepEnsemble.from_manifest(path, device=device)
        if manifest.get("schema_version") != 1 or manifest.get("labels") != {"0": "fake", "1": "confiavel"}:
            raise ValueError("Manifesto do ensemble incompatível.")
        weights = validate_weights(manifest["weights"])
        root = path.resolve().parent

        def artifact_path(relative):
            result = (root / relative).resolve()
            if not result.is_relative_to(root):
                raise ValueError("Artefatos devem pertencer ao diretório do ensemble.")
            return result

        lexical_path = artifact_path(manifest["lexical_path"])
        semantic_path = artifact_path(manifest["transformer_path"])
        signature = manifest.get("split_signature")
        if not signature:
            raise ValueError("Manifesto sem assinatura dos splits de treinamento.")
        for metadata_path in (root / "lexical_metadata.json", semantic_path / "training_metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("split_signature") != signature or metadata.get("labels") != manifest["labels"]:
                raise ValueError("Artefatos com dados, rótulos ou splits incompatíveis.")
        lexical = LexicalClassifier(joblib.load(lexical_path))
        semantic = TransformerClassifier(semantic_path, device=device)
        return cls(lexical, semantic, weights)

    def component_probabilities(self, texts):
        texts = list(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Informe textos não vazios.")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ensemble") as pool:
            lexical = pool.submit(self.lexical.predict_proba, texts)
            semantic = pool.submit(self.semantic.predict_proba, texts)
            return (validate_probabilities(lexical.result(), len(texts)),
                    validate_probabilities(semantic.result(), len(texts)))

    def predict_proba(self, texts):
        lexical, semantic = self.component_probabilities(texts)
        return lexical * self.weights[0] + semantic * self.weights[1]

    def predict(self, text):
        """Recebe string crua; empate no limiar 0.5 favorece confiável, como a API."""
        if not isinstance(text, str):
            raise ValueError("A entrada deve ser uma string.")
        lexical, semantic = self.component_probabilities([text])
        combined = lexical[0] * self.weights[0] + semantic[0] * self.weights[1]
        return {
            "model_type": "SoftVoting (TF-IDF + LogReg + Transformer)",
            "predicted_class": int(combined[1] >= 0.5),
            "probabilities": {"fake": float(combined[0]), "confiavel": float(combined[1])},
            "component_probabilities": {
                "lexical": {"fake": float(lexical[0, 0]), "confiavel": float(lexical[0, 1])},
                "transformer": {"fake": float(semantic[0, 0]), "confiavel": float(semantic[0, 1])},
            },
            "weights": {"lexical": float(self.weights[0]), "transformer": float(self.weights[1])},
            "ml_confidence_percent": float(combined[1] * 100),
        }
