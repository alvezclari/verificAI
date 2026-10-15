"""Valida a aplicação e os modelos reais em CPU, usando uma notícia sintética."""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_model import check_bundle


def main():
    os.environ["MODEL_BACKEND"] = "ensemble"
    os.environ["TRANSFORMER_DEVICE"] = "cpu"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    check_bundle(ROOT, verify_hashes=True)

    from requests import Response
    from backend import create_app, routes

    html = """<!doctype html><html lang="pt-BR"><head>
    <title>Artigo sintético para testar a aplicação</title></head><body><article>
    <h1>Artigo sintético para testar a aplicação</h1>
    <p>Este texto foi criado apenas para validar a execução local do sistema.
    A prefeitura apresentou um relatório sobre melhorias nas bibliotecas municipais.
    O documento descreve atividades de leitura, aquisição de livros e manutenção
    dos espaços. A equipe informou que as etapas serão acompanhadas e os resultados
    divulgados em relatório público. Moradores poderão consultar o calendário de
    atendimento e enviar sugestões ao setor responsável. O conteúdo é sintético
    e não representa uma notícia factual ou uma avaliação de acurácia do modelo.</p>
    </article></body></html>"""
    page = Response()
    page.status_code = 200
    page.url = "https://example.com/verificai-teste"
    page.encoding = "utf-8"
    page._content = html.encode("utf-8")

    with create_app().test_client() as client:
        for endpoint in ("/", "/style.css", "/app.js", "/assets/logo-icone.png", "/docs", "/api/openapi.json"):
            response = client.get(endpoint)
            if response.status_code != 200:
                raise RuntimeError(f"Falha ao servir {endpoint}: HTTP {response.status_code}.")
        health = client.get("/api/health").get_json()
        if (health["status"] != "online" or not health["ensemble_loaded"]
                or not health["model"]["transformer"]["loaded"]):
            raise RuntimeError(f"Modelos indisponíveis: {health.get('model_error')}.")
        # Só a resposta HTTP da página é controlada. Extração e modelos são reais.
        with patch("backend.services.scraper.requests.get", return_value=page):
            response = client.post("/api/analyze", json={"url": page.url})
        data = response.get_json()
        if response.status_code != 200:
            raise RuntimeError(f"Falha na análise: HTTP {response.status_code}: {data}.")
        if (set(data["ml_details"]["component_probabilities"]) != {"lexical", "transformer"}
                or not 0 <= data["prob_fake"] <= 1
                or data["faixa"] not in {"verdadeira", "inconclusiva", "falsa"}
                or data["metadata"]["word_count"] < 40):
            raise RuntimeError("Resposta de classificação incompleta ou inválida.")
        if os.getenv("XAI_ENABLED", "true").lower() in ("true", "1"):
            local = data["ml_details"]["explicacao"].get("local", {})
            if local.get("status") not in {"ok", "no_effect", "insufficient_text"}:
                raise RuntimeError("Explicação local indisponível no teste de inferência.")
    print(json.dumps({
        "status": "ok", "model_version": data["model_version"],
        "device": "cpu", "interface": "ok", "api": "ok", "extraction": "ok",
        "classification": "ok", "components": ["lexical", "transformer"],
        "input": "synthetic_html", "external_downloads": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        sys.exit(f"Validação falhou: {exc}")
