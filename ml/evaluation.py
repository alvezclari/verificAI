"""Métricas e gráficos calculados sobre predições congeladas."""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def bins_confiabilidade(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if n_bins <= 0:
        raise ValueError("Número de bins deve ser positivo.")
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Predições vazias, não finitas ou incompatíveis.")
    if ((p < 0) | (p > 1)).any() or not set(y).issubset({0, 1}):
        raise ValueError("Probabilidades ou rótulos inválidos.")
    indices = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    for i in range(n_bins):
        mascara = indices == i
        bins.append(
            {
                "inicio": i / n_bins,
                "fim": (i + 1) / n_bins,
                "n": int(mascara.sum()),
                "probabilidade_media": float(p[mascara].mean()) if mascara.any() else None,
                "frequencia_verdadeiras": float(y[mascara].mean()) if mascara.any() else None,
            }
        )
    return bins


def metricas(y, p):
    """p é P(verdadeira); FPR/FNR tratam FALSA como classe positiva."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if set(y) != {0, 1}:
        raise ValueError("Avaliação exige ambas as classes.")
    bins = bins_confiabilidade(y, p)
    pred = (p >= 0.5).astype(int)
    matriz = confusion_matrix(1 - y, 1 - pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()
    return {
        "n": len(y),
        "acuracia": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "matriz_confusao": matriz.tolist(),
        "ordem_matriz": ["verdadeira", "falsa"],
        "classe_positiva": "falsa",
        "fpr": float(fp / (fp + tn)),
        "fnr": float(fn / (fn + tp)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece_10_bins": float(
            sum(
                b["n"] / len(y) * abs(b["probabilidade_media"] - b["frequencia_verdadeiras"])
                for b in bins
                if b["n"]
            )
        ),
        "bins": bins,
        "mediana_confiabilidade_verdadeiras": float(np.median(p[y == 1])),
        "mediana_confiabilidade_falsas": float(np.median(p[y == 0])),
        "scores_extremos": int(((p == 0) | (p == 1)).sum()),
        "taxas_arredondadas_extremas": int(np.isin(np.round(p * 100, 2), [0, 100]).sum()),
    }


def avaliar_particao(artefato, dados, destino: Path, nome: str):
    if dados.empty:
        return None
    previsoes = artefato.prever_textos(dados.texto.tolist())
    y = dados.label.to_numpy()
    resultado = {etapa: metricas(y, previsoes[etapa]) for etapa in ("bruto", "calibrado", "final")}
    por_fonte = {}
    for fonte in sorted(dados.fonte.unique()):
        mascara = dados.fonte.eq(fonte).to_numpy()
        yf, pf = y[mascara], previsoes["final"][mascara]
        bins = bins_confiabilidade(yf, pf)
        por_fonte[fonte] = {
            "n": int(mascara.sum()),
            "falsas": int((yf == 0).sum()),
            "verdadeiras": int((yf == 1).sum()),
            "acuracia": float(accuracy_score(yf, pf >= 0.5)),
            "log_loss": float(log_loss(yf, pf, labels=[0, 1])),
            "brier": float(brier_score_loss(yf, pf)),
            "ece_10_bins": float(
                sum(
                    b["n"] / len(yf) * abs(b["probabilidade_media"] - b["frequencia_verdadeiras"])
                    for b in bins
                    if b["n"]
                )
            ),
            "fonte_no_desenvolvimento": fonte
            in artefato.manifest.get("fontes_desenvolvimento", []),
        }
    resultado["por_fonte"] = por_fonte
    destino.mkdir(parents=True, exist_ok=True)
    tabela = dados[["id", "label", "fonte", "grupo"]].copy()
    for etapa in ("bruto", "calibrado", "final"):
        tabela[f"prob_verdadeira_{etapa}"] = previsoes[etapa]
    tabela.to_csv(destino / f"{nome}_predicoes.csv", index=False)
    figura, eixos = plt.subplots(1, 2, figsize=(11, 4))
    eixos[0].plot([0, 1], [0, 1], "--", color="gray")
    for etapa in ("bruto", "calibrado", "final"):
        bins = [b for b in resultado[etapa]["bins"] if b["n"]]
        eixos[0].plot(
            [b["probabilidade_media"] for b in bins],
            [b["frequencia_verdadeiras"] for b in bins],
            "o-",
            label=etapa,
        )
    eixos[0].set(
        xlabel="Confiabilidade prevista",
        ylabel="Frequência de verdadeiras",
        xlim=(0, 1),
        ylim=(0, 1),
        title=f"Confiabilidade — {nome}",
    )
    eixos[0].legend()
    for classe in (0, 1):
        eixos[1].hist(
            previsoes["final"][y == classe] * 100,
            bins=np.linspace(0, 100, 21),
            alpha=0.6,
            label="falsa" if classe == 0 else "verdadeira",
        )
    eixos[1].set(xlabel="Taxa de confiabilidade (%)", ylabel="Notícias", title="Classes reais")
    eixos[1].legend()
    figura.tight_layout()
    figura.savefig(destino / f"{nome}_confiabilidade.svg")
    plt.close(figura)
    return resultado


def criterios_aceite(resultado):
    final = resultado["final"]
    return {
        "ece_ate_005": final["ece_10_bins"] <= 0.05,
        "sem_extremos": final["scores_extremos"] == final["taxas_arredondadas_extremas"] == 0,
        "mediana_verdadeiras_acima_065": final["mediana_confiabilidade_verdadeiras"] > 0.65,
        "mediana_falsas_abaixo_035": final["mediana_confiabilidade_falsas"] < 0.35,
    }


def casos_sanidade(artefato):
    casos = {
        "numero_82": ("Casos confirmados", "Foram registrados 82 casos no município."),
        "numero_8200": ("Casos confirmados", "Foram registrados 8200 casos no município."),
        "com_negacao": ("Comunicado", "A prefeitura não confirmou o aumento dos casos."),
        "sem_negacao": ("Comunicado", "A prefeitura confirmou o aumento dos casos."),
        "curto_vago": ("", "Algo aconteceu hoje."),
    }
    return {
        nome: {"prob_fake": float(1 - artefato.prever_documento(t, c)["final"][0])}
        for nome, (t, c) in casos.items()
    }
