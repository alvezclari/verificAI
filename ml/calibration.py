"""Calibração monotônica ajustada somente na reserva de calibração."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from ml.dataset import ProtocoloInviavel, dobras_agrupadas


@dataclass
class Calibrador:
    metodo: str
    inclinacao: float = 1.0
    intercepto: float = 0.0
    isotonico: object = None
    ajustado: bool = False

    def ajustar(self, margens, rotulos):
        x, y = np.asarray(margens, dtype=float), np.asarray(rotulos, dtype=int)
        if len(x) != len(y) or len(x) == 0 or not np.isfinite(x).all() or set(y) != {0, 1}:
            raise ValueError("Calibração exige margens finitas e ambas as classes.")
        if self.metodo == "sigmoid":
            # A inclinação não negativa impede inversão da polaridade aprendida.
            def objetivo(parametros):
                logits = parametros[0] * x + parametros[1]
                erro = expit(logits) - y
                perda = np.mean(np.logaddexp(0, logits) - y * logits)
                gradiente = np.array([np.mean(erro * x), np.mean(erro)])
                return perda, gradiente

            resultado = minimize(
                objetivo, [1.0, 0.0], jac=True, method="L-BFGS-B", bounds=[(0, None), (None, None)]
            )
            if not resultado.success:
                raise ValueError(f"Falha ao ajustar sigmoid: {resultado.message}")
            self.inclinacao, self.intercepto = map(float, resultado.x)
        elif self.metodo == "temperature":

            def objetivo(log_temperatura):
                logits = x / np.exp(log_temperatura)
                return np.mean(np.logaddexp(0, logits) - y * logits)

            # Limites numéricos da otimização, não limites fixos de probabilidades.
            resultado = minimize_scalar(objetivo, bounds=(-8, 8), method="bounded")
            if not resultado.success:
                raise ValueError("Falha ao ajustar a temperatura.")
            self.inclinacao = float(np.exp(-resultado.x))
            self.intercepto = 0.0
        elif self.metodo == "isotonica":
            if len(y) < 1000:
                raise ValueError("Isotônica exige ao menos 1.000 exemplos por ajuste.")
            self.isotonico = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
        else:
            raise ValueError(f"Método de calibração desconhecido: {self.metodo}")
        self.ajustado = True
        return self

    def prever(self, margens):
        if not self.ajustado:
            raise ValueError("Calibrador ainda não ajustado.")
        margens = np.asarray(margens, dtype=float)
        if not np.isfinite(margens).all():
            raise ValueError("Margens não finitas.")
        if self.metodo == "isotonica":
            if self.isotonico is None:
                raise ValueError("Calibrador isotônico ainda não ajustado.")
            return self.isotonico.predict(margens)
        return expit(self.inclinacao * margens + self.intercepto)


def contrair(probabilidades, r: float):
    """Contrai em direção a 0,5 com coeficiente documentado e estritamente positivo."""
    p = np.asarray(probabilidades, dtype=float)
    if not 0 < r <= 1:
        raise ValueError("r deve estar em (0, 1].")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilidades devem estar em [0, 1].")
    return 0.5 + (p - 0.5) * r


def escolher_calibrador(margens, dados_calibracao) -> tuple:
    """Seleciona por predições OOF agrupadas; teste não participa da seleção."""
    margens = np.asarray(margens)
    y = dados_calibracao.label.to_numpy()
    dobras = dobras_agrupadas(dados_calibracao)
    metodos = ["temperature", "sigmoid"]
    if min(len(treino) for treino, _ in dobras) >= 1000:
        metodos.append("isotonica")
    comparacao, previsoes = {}, {}
    for metodo in metodos:
        oof = np.full(len(y), np.nan)
        for treino, validacao in dobras:
            cal = Calibrador(metodo).ajustar(margens[treino], y[treino])
            oof[validacao] = cal.prever(margens[validacao])
        if not np.isfinite(oof).all():
            raise AssertionError("Predições OOF incompletas na calibração.")
        comparacao[metodo] = {
            "log_loss": float(log_loss(y, oof, labels=[0, 1])),
            "brier": float(brier_score_loss(y, oof)),
        }
        previsoes[metodo] = oof
    # Empates numéricos favorecem temperatura, depois sigmoid, depois isotônica.
    melhor = metodos[0]
    for metodo in metodos[1:]:
        a, b = comparacao[metodo], comparacao[melhor]
        if a["log_loss"] < b["log_loss"] - 1e-4 or (
            abs(a["log_loss"] - b["log_loss"]) <= 1e-4 and a["brier"] < b["brier"] - 1e-4
        ):
            melhor = metodo
    oof = previsoes[melhor]
    acertos = int(np.sum((oof >= 0.5).astype(int) == y))
    # Estimativa Beta(1,1): evita certeza absoluta com amostra finita.
    acuracia_suavizada = (acertos + 1) / (len(y) + 2)
    r = 2 * acuracia_suavizada - 1
    if r <= 0:
        raise ProtocoloInviavel(
            "Calibração OOF não supera o acaso; não é possível estimar r positivo."
        )
    detalhes = {
        "comparacao_oof_calibracao": comparacao,
        "metodo": melhor,
        "isotonica_omitida": "isotonica" not in metodos,
        "r": r,
        "origem_r": "2 * ((acertos_oof_calibracao + 1) / (n_calibracao + 2)) - 1",
        "prior_acuracia": "Beta(1,1)",
        "acertos_oof_calibracao": acertos,
        "n_calibracao": len(y),
        "acuracia_oof_calibracao": float(accuracy_score(y, oof >= 0.5)),
        "teste_usado_para_ajuste": False,
    }
    return Calibrador(melhor).ajustar(margens, y), r, detalhes
