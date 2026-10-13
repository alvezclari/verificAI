"""Treinamento agrupado com calibração independente e teste final intocado."""

import argparse
import hashlib
import importlib.metadata
import platform
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

from ml.artifact import VERSAO_MANIFEST, ArtefatoClassificador
from ml.calibration import escolher_calibrador
from ml.dataset import (
    ProtocoloInviavel,
    agrupar_quase_duplicatas,
    carregar_dataset,
    deduplicar,
    dobras_agrupadas,
    excluir_conflitos_url,
    exportar_pendencias,
    hash_arquivo,
    particionar,
    salvar_json,
    validar_viabilidade,
)
from ml.evaluation import avaliar_particao, casos_sanidade, criterios_aceite


def construir_pipeline(seed=42):
    return Pipeline(
        [
            (
                "atributos",
                FeatureUnion(
                    [
                        (
                            "palavras",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                max_features=20000,
                                lowercase=False,
                                token_pattern=r"(?u)r\$|%|\b\w+\b",
                            ),
                        ),
                        (
                            "caracteres",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(3, 5),
                                max_features=30000,
                                lowercase=False,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("classificador", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


def treinar(caminho, destino, pasta_modelo, seed=42, fracao_temporal=0.15, protocolo="fonte"):
    destino, pasta_modelo = Path(destino), Path(pasta_modelo)
    dados, auditoria = carregar_dataset(caminho)
    if protocolo not in {"fonte", "experimental"}:
        raise ValueError("Protocolo deve ser fonte ou experimental.")
    auditoria["protocolo"] = protocolo
    exportar_pendencias(dados, destino)
    exclusoes = None
    if protocolo == "experimental":
        dados, exclusoes = excluir_conflitos_url(dados, destino)
        auditoria["exclusoes_conflitos"] = exclusoes
        auditoria["limitacoes"].append(
            "Protocolo experimental autorizado: publicadores podem aparecer em várias partições."
        )
    try:
        print("Preparando grupos e verificando quase-duplicatas...", flush=True)
        dados = deduplicar(dados, agrupar_por_fonte=protocolo == "fonte")
        auditoria["linhas_apos_deduplicacao"] = len(dados)
        auditoria["viabilidade"] = validar_viabilidade(dados)
        if not auditoria["viabilidade"]["viavel"]:
            raise ProtocoloInviavel(" ".join(auditoria["viabilidade"]["motivos"]))
        dados, similaridade = agrupar_quase_duplicatas(dados)
        auditoria["quase_duplicatas"] = similaridade
        partes = particionar(dados, seed, fracao_temporal)
        partes.salvar(destino / "particoes.json")
        print("Partições validadas; iniciando busca de hiperparâmetros...", flush=True)
        busca = GridSearchCV(
            construir_pipeline(seed),
            {"classificador__C": [0.5, 2.0]},
            cv=dobras_agrupadas(partes.treino),
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
            refit=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            busca.fit(partes.treino.texto, partes.treino.label)
        pipeline = busca.best_estimator_
        print("Treino concluído; ajustando calibração na reserva independente...", flush=True)
        margens_cal = pipeline.decision_function(partes.calibracao.texto)
        calibrador, r, detalhes = escolher_calibrador(margens_cal, partes.calibracao)
    except ProtocoloInviavel as erro:
        auditoria["status"] = "bloqueada_por_protocolo"
        auditoria["impedimento"] = str(erro)
        salvar_json(destino / "auditoria_dados.json", auditoria)
        salvar_json(
            destino / "metrics.json",
            {
                "status": "nao_avaliado",
                "motivo": str(erro),
                "teste": None,
                "temporal": None,
                "urls_teste": None,
                "artefato_gerado_nesta_execucao": False,
            },
        )
        raise
    versao = f"texto-{auditoria['sha256'][:12]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    manifest = {
        "schema_version": VERSAO_MANIFEST,
        "versao_modelo": versao,
        "dataset_sha256": auditoria["sha256"],
        "dataset": str(caminho),
        "rotulos": {"0": "falsa", "1": "verdadeira"},
        "classes": pipeline.classes_.tolist(),
        "entrada": "titulo_e_corpo",
        "politica_dados": "corpus_transformado_exploratorio",
        "protocolo": protocolo,
        "agrupar_por_fonte": protocolo == "fonte",
        "exclusoes_conflitos": exclusoes,
        "preprocessamento": "ml.preprocessing.preparar_texto",
        "preprocessamento_sha256": hashlib.sha256(
            Path(__file__).with_name("preprocessing.py").read_bytes()
        ).hexdigest(),
        "bibliotecas": {
            nome: importlib.metadata.version(nome)
            for nome in ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "tldextract")
        },
        "python": platform.python_version(),
        "seed": seed,
        "hiperparametros": busca.best_params_,
        "f1_macro_cv_treino": float(busca.best_score_),
        "configuracao_tfidf": {
            "palavras": {"ngramas": [1, 2], "max_features": 20000},
            "caracteres": {"ngramas": [3, 5], "max_features": 30000},
        },
        "tamanhos": {
            nome: len(getattr(partes, nome))
            for nome in ("treino", "calibracao", "teste", "temporal", "excluidos_temporal")
        },
        "proporcoes_alvo": [0.70, 0.15, 0.15],
        "fontes_desenvolvimento": sorted(set(partes.treino.fonte) | set(partes.calibracao.fonte)),
        "fontes_compartilhadas_treino_teste": sorted(
            set(partes.treino.fonte) & set(partes.teste.fonte)
        ),
        "corte_temporal": partes.corte_temporal,
        "particoes_sha256": hash_arquivo(destino / "particoes.json"),
        "similaridade": similaridade,
        "calibracao": detalhes,
        "r": r,
        "limiares_prob_fake": {"verdadeira_max": 0.35, "inconclusiva_max": 0.65},
        "limiares_status": "predefinidos; avaliação final não os ajusta",
        "limitacoes": auditoria["limitacoes"],
    }
    minimo = max(1, int(np.quantile(partes.treino.texto.str.split().str.len(), 0.05)))
    manifest["ood_texto_curto"] = {
        "minimo_palavras": minimo,
        "origem": "percentil_5_treino",
        "fator": "min(1, palavras_entrada / minimo_palavras)",
    }
    artefato = ArtefatoClassificador(pipeline, calibrador, r, minimo, manifest)
    teste = avaliar_particao(artefato, partes.teste, destino, "teste")
    temporal = avaliar_particao(artefato, partes.temporal, destino, "temporal")
    metricas = {
        "teste": teste,
        "temporal": temporal,
        "sanidade": casos_sanidade(artefato),
        "urls_teste": None,
        "motivo_urls": "URLs rotuladas não fornecidas.",
        "aceite_teste": criterios_aceite(teste),
        "aceite_temporal": criterios_aceite(temporal) if temporal else None,
    }
    manifest["criterios_aceite"] = metricas["aceite_teste"]
    manifest["apto_para_producao"] = False
    manifest["motivo_producao"] = "Ausência de validação independente com corpos originais."
    pasta_modelo.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefato, pasta_modelo / "classificador.joblib")
    salvar_json(pasta_modelo / "manifest.json", manifest)
    salvar_json(destino / "metrics.json", metricas)
    auditoria["status"] = "avaliada_exploratoriamente"
    salvar_json(destino / "auditoria_dados.json", auditoria)
    return artefato, metricas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/fake_recogna_limpo.csv")
    parser.add_argument("--saida", default=f"reports/{date.today().isoformat()}")
    parser.add_argument("--modelo", default="models/textual_calibrado")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fracao-temporal", type=float, default=0.15)
    parser.add_argument("--protocolo", choices=["fonte", "experimental"], default="fonte")
    args = parser.parse_args()
    try:
        _, resultado = treinar(
            args.dataset, args.saida, args.modelo, args.seed, args.fracao_temporal, args.protocolo
        )
    except ProtocoloInviavel as erro:
        print(f"Treinamento interrompido: {erro}")
        print(f"Diagnóstico em {args.saida}/auditoria_dados.json; métricas indisponíveis.")
        return 2
    aprovado = all(resultado["aceite_teste"].values())
    if resultado["aceite_temporal"]:
        aprovado = aprovado and all(resultado["aceite_temporal"].values())
    print(f"Avaliação concluída. Critérios quantitativos atendidos: {aprovado}")
    return 0 if aprovado else 2


if __name__ == "__main__":
    raise SystemExit(main())
