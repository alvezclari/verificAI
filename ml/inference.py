"""Motor de inferência local, sem downloads nem scores substitutos."""

import importlib.metadata
import math
from pathlib import Path

import joblib
import numpy as np

from ml.artifact import VERSAO_MANIFEST, ArtefatoClassificador
from ml.dataset import hash_arquivo
from ml.preprocessing import preparar_texto


class ModeloIndisponivel(RuntimeError):
    """O artefato não pôde ser validado e nenhuma probabilidade pode ser emitida."""


class MotorClassificador:
    def __init__(self, caminho: str | Path):
        self.ready = False
        self.load_error = None
        self.artefato = None
        self.versao_modelo = None
        try:
            artefato = joblib.load(caminho)
            if not isinstance(artefato, ArtefatoClassificador):
                raise ValueError("Tipo de artefato incompatível.")
            manifest = artefato.manifest
            if manifest.get("schema_version") != VERSAO_MANIFEST:
                raise ValueError("Versão do manifesto incompatível.")
            if manifest.get("rotulos") != {"0": "falsa", "1": "verdadeira"}:
                raise ValueError("Mapeamento de rótulos incompatível.")
            if list(artefato.pipeline.classes_) != [0, 1] or manifest.get("classes") != [0, 1]:
                raise ValueError("Classes incompatíveis.")
            if manifest.get("entrada") != "titulo_e_corpo":
                raise ValueError("Tipo de texto de entrada incompatível.")
            if manifest.get("preprocessamento_sha256") != hash_arquivo(
                Path(__file__).with_name("preprocessing.py")
            ):
                raise ValueError("Pré-processamento diverge do usado no treino.")
            if manifest.get("bibliotecas", {}).get("scikit-learn") != importlib.metadata.version(
                "scikit-learn"
            ):
                raise ValueError("Versão do scikit-learn diverge do treino.")
            if not math.isfinite(artefato.r) or not 0 < artefato.r <= 1:
                raise ValueError("Coeficiente de contração inválido.")
            if manifest.get("r") != artefato.r or artefato.minimo_palavras < 1:
                raise ValueError("Metadados de contração incompatíveis.")
            if not artefato.calibrador.ajustado:
                raise ValueError("Calibrador não ajustado.")
            metodo = artefato.calibrador.metodo
            if metodo not in {"sigmoid", "temperature", "isotonica"} or (
                manifest.get("calibracao", {}).get("metodo") != metodo
            ):
                raise ValueError("Método de calibração incompatível.")
            if not math.isfinite(artefato.calibrador.inclinacao) or (
                artefato.calibrador.inclinacao < 0
            ):
                raise ValueError("Polaridade do calibrador incompatível.")
            if not isinstance(manifest.get("versao_modelo"), str) or not manifest["versao_modelo"]:
                raise ValueError("Versão do modelo ausente.")
            self.artefato = artefato
            self.versao_modelo = manifest["versao_modelo"]
            self.ready = True
        except Exception as erro:
            self.load_error = f"Modelo indisponível: {type(erro).__name__}: {erro}"

    def prever(self, titulo: str | None, corpo: str | None) -> dict:
        if not self.ready:
            raise ModeloIndisponivel(
                "Classificador indisponível; consulte o diagnóstico do serviço."
            )
        texto = preparar_texto(titulo, corpo)
        valores = self.artefato.prever_textos([texto])
        prob_fake = float(1 - valores["final"][0])
        limites = self.artefato.manifest["limiares_prob_fake"]
        faixa = (
            "verdadeira"
            if prob_fake <= limites["verdadeira_max"]
            else ("inconclusiva" if prob_fake <= limites["inconclusiva_max"] else "falsa")
        )
        atributos = self.artefato.pipeline.named_steps["atributos"]
        vetor = atributos.transform([texto]).tocsr()
        coeficientes = self.artefato.pipeline.named_steps["classificador"].coef_[0]
        nomes = atributos.get_feature_names_out()
        contribuicoes = []
        for i, valor in zip(vetor.indices, vetor.data):
            nome = nomes[i]
            if nome.startswith("palavras__"):
                contribuicoes.append(
                    {
                        "termo": nome.removeprefix("palavras__"),
                        "contribuicao_logit_verdadeira": float(valor * coeficientes[i]),
                    }
                )
        sustenta = sorted(
            (t for t in contribuicoes if t["contribuicao_logit_verdadeira"] > 0),
            key=lambda t: t["contribuicao_logit_verdadeira"],
            reverse=True,
        )[:5]
        enfraquece = sorted(
            (t for t in contribuicoes if t["contribuicao_logit_verdadeira"] < 0),
            key=lambda t: t["contribuicao_logit_verdadeira"],
        )[:5]
        if not np.isfinite(prob_fake) or not 0 < prob_fake < 1:
            raise ModeloIndisponivel("O classificador produziu uma probabilidade inválida.")
        return {
            "prob_fake": prob_fake,
            "prob_fake_bruto": float(1 - valores["bruto"][0]),
            "metodo_calibracao": self.artefato.calibrador.metodo,
            "versao_modelo": self.versao_modelo,
            "faixa": faixa,
            "explicacao": {
                "sustenta": sustenta,
                "enfraquece": enfraquece,
                "limitacoes": [
                    "Contribuições textuais, não evidências factuais.",
                    "N-gramas de caracteres e intercepto também participam.",
                ],
            },
            "fator_texto_curto": float(valores["fator_texto_curto"][0]),
        }
