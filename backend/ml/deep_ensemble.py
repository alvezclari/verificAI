"""Ensemble do artefato calibrado importado e do mDeBERTa supervisionado."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from backend.ml.calibrated import CalibratedClassifier
from backend.ml.ensemble import softmax, validate_weights, validate_probabilities
from backend.ml.transformer import TransformerClassifier


def calibrated_blend(lexical, semantic, weights, temperature=1.0, bias=0.0):
    weights = validate_weights(weights)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperatura do ensemble inválida.")
    if not np.isfinite(bias):
        raise ValueError("Intercepto de calibração inválido.")
    lexical, semantic = validate_probabilities(lexical), validate_probabilities(semantic)
    if lexical.shape != semantic.shape:
        raise ValueError("Componentes retornaram quantidades diferentes de predições.")
    blended = lexical * weights[0] + semantic * weights[1]
    logits = np.log(np.clip(blended, 1e-12, 1)) / temperature
    logits[:, 0] += bias
    return softmax(logits)


class CalibratedDeepEnsemble:
    def __init__(self, lexical, semantic, weights=(0.5, 0.5), temperature=1.0, manifest=None, bias=0.0):
        self.lexical, self.semantic = lexical, semantic
        self.weights = validate_weights(weights)
        self.temperature = float(temperature)
        self.bias = float(bias)
        if not np.isfinite(self.bias):
            raise ValueError("Intercepto de calibração inválido.")
        self.manifest = manifest or {}
        self.thresholds = self.manifest.get("thresholds", {"verdadeira_max": 0.35, "inconclusiva_max": 0.65})
        if not (0 <= self.thresholds["verdadeira_max"] < self.thresholds["inconclusiva_max"] <= 1):
            raise ValueError("Faixas do ensemble inválidas.")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperatura do ensemble inválida.")
        self.version = self.manifest.get("version", "experimental")

    @classmethod
    def from_manifest(cls, path, device="cpu"):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2 or data.get("labels") != {"0": "fake", "1": "confiavel"}:
            raise ValueError("Manifesto de ensemble calibrado incompatível.")
        def resolve(value):
            target = (path.parent / value).resolve()
            if not target.is_relative_to(path.parent):
                raise ValueError("Artefato fora do diretório do bundle.")
            return target
        lexical = CalibratedClassifier(resolve(data["calibrated_path"]))
        if lexical.sha256 != data["calibrated_sha256"]:
            raise ValueError("Hash do classificador importado diverge do manifesto.")
        semantic = TransformerClassifier(resolve(data["transformer_path"]), device=device)
        if semantic.metadata.get("split_signature") != data.get("split_signature") or not data.get("split_signature"):
            raise ValueError("Splits do Transformer divergem do bundle.")
        if semantic.metadata.get("task") != "supervised_fake_news":
            raise ValueError("Transformer não foi ajustado para classificação de notícias.")
        result = cls(lexical, semantic, data["weights"], data["temperature"], data, bias=data.get("bias", 0.0))
        result.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def component_probabilities(self, texts):
        texts = list(texts)
        with ThreadPoolExecutor(max_workers=2) as pool:
            lexical = pool.submit(self.lexical.predict_proba, texts)
            semantic = pool.submit(self.semantic.predict_proba, texts)
            return lexical.result(), semantic.result()

    def predict_proba(self, texts):
        lexical, semantic = self.component_probabilities(texts)
        return calibrated_blend(lexical, semantic, self.weights, self.temperature, self.bias)

    def predict_documents_proba(self, documents):
        """Avalia perturbações com a calibração final e lotes pequenos em memória."""
        documents = list(documents)
        if not documents:
            raise ValueError("Informe documentos não vazios.")
        results = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0, len(documents), 2):
                batch = documents[start:start + 2]
                texts = [f"{title or ''}\n\n{body or ''}".strip() for title, body in batch]
                lexical = pool.submit(self.lexical.predict_documents_proba, batch)
                semantic = pool.submit(self.semantic.predict_proba, texts)
                results.append(calibrated_blend(
                    lexical.result(), semantic.result(), self.weights, self.temperature, self.bias,
                ))
        return np.concatenate(results)

    def predict_document(self, title, body):
        content = f"{title or ''}\n\n{body or ''}".strip()
        with ThreadPoolExecutor(max_workers=2) as pool:
            lexical_future = pool.submit(self.lexical.predict_document, title, body)
            semantic_future = pool.submit(self.semantic.predict_proba, [content])
            lexical_result = lexical_future.result()
            lexical_probs = np.array([[lexical_result["prob_fake"], 1 - lexical_result["prob_fake"]]])
            semantic_probs = semantic_future.result()
        probabilities = calibrated_blend(lexical_probs, semantic_probs, self.weights, self.temperature, self.bias)[0]
        fake = float(probabilities[0])
        band = "verdadeira" if fake <= self.thresholds["verdadeira_max"] else ("inconclusiva" if fake <= self.thresholds["inconclusiva_max"] else "falsa")
        input_info = self.semantic.input_info(content) if hasattr(self.semantic, "input_info") else {}
        return {
            "model_type": "Ensemble LogReg calibrada + mDeBERTa-v3",
            "prob_fake": fake, "probabilities": {"fake": fake, "confiavel": float(probabilities[1])},
            "ml_confidence_percent": float(probabilities[1] * 100), "faixa": band,
            "versao_modelo": self.version, "model_sha256": getattr(self, "sha256", None),
            "thresholds": self.thresholds,
            "component_probabilities": {
                "lexical": {"fake": float(lexical_probs[0, 0]), "confiavel": float(lexical_probs[0, 1])},
                "transformer": {"fake": float(semantic_probs[0, 0]), "confiavel": float(semantic_probs[0, 1])},
            },
            "weights": {"lexical": float(self.weights[0]), "transformer": float(self.weights[1])},
            "metodo_calibracao": self.manifest.get("calibration_method", "temperature_on_reserved_calibration_split"),
            "temperature": self.temperature, "bias": self.bias, "fator_texto_curto": lexical_result["fator_texto_curto"],
            "explicacao": {**lexical_result["explicacao"], "scope": "lexical_component_only"},
            "transformer": {
                **input_info,
                "base_model": self.semantic.metadata.get("source_model", self.semantic.metadata.get("base_model")),
                "task": self.semantic.metadata.get("task"), "max_length": self.semantic.max_length,
                "device": str(self.semantic.device), "temperature": self.semantic.temperature,
                "training_run_id": self.semantic.metadata.get("run_id"),
                "text_policy": "Primeiros max_length tokens; notícias longas são truncadas.",
            },
            "evaluation": self.manifest.get("evaluation", {}),
            "limitations": self.manifest.get("limitations", []),
        }
