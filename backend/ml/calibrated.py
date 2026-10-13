"""Adaptador do artefato importado, preservando todas as regras originais."""

import hashlib
from pathlib import Path

import numpy as np

from backend.ml.errors import ModelUnavailableError


class CalibratedClassifier:
    def __init__(self, path):
        from ml.inference import MotorClassificador

        self.path = Path(path)
        self.motor = MotorClassificador(self.path)
        if not self.motor.ready:
            raise ModelUnavailableError(self.motor.load_error)
        self.manifest = self.motor.artefato.manifest
        limits = self.manifest.get("limiares_prob_fake", {})
        if not (0 <= limits.get("verdadeira_max", -1) < limits.get("inconclusiva_max", -1) <= 1):
            raise ModelUnavailableError("Limiares do artefato inválidos.")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def version(self):
        return self.motor.versao_modelo

    def predict_proba(self, texts):
        from ml.preprocessing import preparar_texto
        from backend.ml.ensemble import validate_probabilities

        texts = list(texts)
        prepared = [preparar_texto(None, text) for text in texts]
        if not prepared or any(not text for text in prepared):
            raise ValueError("Informe textos não vazios.")
        results = []
        for start in range(0, len(prepared), 64):
            reliable = self.motor.artefato.prever_textos(prepared[start:start + 64])["final"]
            results.append(np.column_stack((1 - reliable, reliable)))
        return validate_probabilities(np.concatenate(results), len(texts))

    def predict_document(self, title, body):
        try:
            result = self.motor.prever(title, body)
        except Exception as exc:
            raise ModelUnavailableError("O classificador não conseguiu analisar o texto extraído.") from exc
        probability = result["prob_fake"]
        return {
            **result,
            "model_type": "TF-IDF (palavras + caracteres) + LogReg calibrada",
            "probabilities": {"fake": probability, "confiavel": 1 - probability},
            "ml_confidence_percent": (1 - probability) * 100,
            "model_sha256": self.sha256,
            "thresholds": self.manifest["limiares_prob_fake"],
            "limitations": self.manifest.get("limitacoes", []),
            "evaluation": {
                "protocol": self.manifest.get("protocolo"),
                "f1_macro_cv_train": self.manifest.get("f1_macro_cv_treino"),
                "calibration_oof": self.manifest.get("calibracao", {}).get("comparacao_oof_calibracao", {}),
                "production_ready": self.manifest.get("apto_para_producao", False),
                "note": "Métricas do corpus de treinamento; não medem o acerto desta URL.",
            },
        }
