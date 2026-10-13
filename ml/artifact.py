"""Artefato único contendo pipeline, calibrador, contração e manifesto."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from ml.calibration import contrair
from ml.preprocessing import preparar_texto

VERSAO_MANIFEST = "1.0"


@dataclass
class ArtefatoClassificador:
    pipeline: object
    calibrador: object
    r: float
    minimo_palavras: int
    manifest: dict

    def prever_textos(self, textos):
        margens = self.pipeline.decision_function(textos)
        bruto = expit(margens)
        calibrado = self.calibrador.prever(margens)
        final = contrair(calibrado, self.r)
        tamanhos = np.array([len(texto.split()) for texto in textos])
        if (tamanhos == 0).any():
            raise ValueError("Texto vazio não pode receber score.")
        # Limite aprendido no treino: percentil 5 do número de palavras.
        fator = np.minimum(1.0, tamanhos / self.minimo_palavras)
        final = 0.5 + (final - 0.5) * fator
        # Apenas proteção de ponto flutuante, sem score factual artificial.
        final = np.clip(final, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        return {"bruto": bruto, "calibrado": calibrado, "final": final, "fator_texto_curto": fator}

    def prever_documento(self, titulo, corpo):
        return self.prever_textos([preparar_texto(titulo, corpo)])
