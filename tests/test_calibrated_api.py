import importlib.metadata
import json
from pathlib import Path

import pytest
import requests

from backend import create_app
from backend.config import Config
from backend.services.analyzer import NewsAnalyzerService

pytestmark = pytest.mark.skipif(
    importlib.metadata.version("scikit-learn") != "1.8.0",
    reason="O artefato real exige requirements-classificador.txt (sklearn 1.8.0).",
)

TITLE = "Ministério da Saúde apresenta balanço da vacinação"
BODY = (
    "O Ministério da Saúde publicou um relatório sobre a campanha de vacinação. "
    "Segundo a nota técnica, os dados foram enviados pelas secretarias estaduais e municipais. "
    "O documento apresenta a metodologia e informa que os resultados podem ser revisados após a atualização dos registros."
)
URL = "https://noticias.exemplo.com.br/vacinacao"


@pytest.fixture
def service(monkeypatch):
    from backend import routes
    monkeypatch.setattr(Config, "MODEL_BACKEND", "calibrated")
    monkeypatch.setattr(Config, "CLASSIFIER_PATH", Path(__file__).resolve().parents[1] / "models/classificador.joblib")
    analyzer = NewsAnalyzerService()
    assert analyzer.is_model_loaded(), analyzer.model_error
    monkeypatch.setattr(routes, "analyzer_service", analyzer)
    return analyzer


@pytest.fixture
def client(service):
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_url_html_extraction_and_real_model_match_original_motor(client, service, monkeypatch):
    from backend import routes
    response = requests.Response()
    response.status_code = 200
    response.url = URL
    response.encoding = "utf-8"
    response._content = f'''<!doctype html><html lang="pt-BR"><head>
      <title>{TITLE}</title><meta name="author" content="Redação Saúde">
      <meta property="article:published_time" content="2026-10-13T10:00:00Z">
      </head><body><article><h1>{TITLE}</h1><p>{BODY}</p></article></body></html>'''.encode()
    downloads = []

    def get(url, **kwargs):
        downloads.append(url)
        return response

    monkeypatch.setattr("backend.services.scraper.requests.get", get)
    article = routes.scraper_service.scrape(URL)
    expected = service.calibrated.motor.prever(article["title"], article["text"])
    result = client.post("/api/analyze", json={"url": URL})
    assert result.status_code == 200
    data = result.get_json()
    assert data["prob_fake"] == expected["prob_fake"]
    assert data["confidence_score"] == pytest.approx((1 - expected["prob_fake"]) * 100)
    assert data["faixa"] == expected["faixa"]
    assert data["ml_details"]["explicacao"] == expected["explicacao"]
    assert data["model_version"] == service.calibrated.version
    assert data["score_method"] == "calibrated_model"
    assert not data["criteria_affect_score"]
    assert data["threshold"] is None and data["thresholds"] == {"verdadeira_max": 0.35, "inconclusiva_max": 0.65}
    assert len(data["provenance"]["input_sha256"]) == 64
    assert data["provenance"]["extraction"]["final_url"] == URL
    assert data["provenance"]["model_sha256"] == service.calibrated.sha256
    again = client.post("/api/analyze", json={"url": URL}).get_json()
    assert again["prob_fake"] == data["prob_fake"]
    assert again["provenance"]["input_sha256"] == data["provenance"]["input_sha256"]
    assert downloads == [URL, URL, URL]


@pytest.mark.parametrize("prob_fake,faixa,verdict", [
    (0.35, "verdadeira", "confiavel"),
    (0.350001, "inconclusiva", "inconclusiva"),
    (0.65, "inconclusiva", "inconclusiva"),
    (0.650001, "falsa", "desconfiavel"),
])
def test_original_probability_bands_are_preserved(client, service, monkeypatch, prob_fake, faixa, verdict):
    # Limiares são exercitados pelo motor real; apenas a saída numérica do artefato é controlada.
    monkeypatch.setattr(service.calibrated.motor.artefato, "prever_textos", lambda texts: {
        "final": [1 - prob_fake], "bruto": [1 - prob_fake], "fator_texto_curto": [1.0],
    })
    monkeypatch.setattr("backend.routes.scraper_service.scrape", lambda url: {"title": TITLE, "text": BODY, "url": url})
    data = client.post("/api/analyze", json={"url": URL}).get_json()
    assert data["faixa"] == faixa
    assert data["verdict"] == verdict
    assert data["confidence_score"] == pytest.approx(100 * (1 - prob_fake))


def test_health_reload_and_missing_model_are_explicit(client, service, monkeypatch, tmp_path):
    health = client.get("/api/health").get_json()
    assert health["status"] == "online"
    assert health["model"]["backend"] == "calibrated"
    assert health["model"]["version"] == service.calibrated.version
    assert client.post("/api/reload-model").status_code == 200
    monkeypatch.setattr(Config, "CLASSIFIER_PATH", tmp_path / "missing.joblib")
    assert client.post("/api/reload-model").status_code == 503
    assert client.get("/api/health").get_json()["status"] == "degraded"
    monkeypatch.setattr("backend.routes.scraper_service.scrape", lambda url: pytest.fail("Não deve acessar rede sem modelo"))
    response = client.post("/api/analyze", json={"url": URL})
    assert response.status_code == 503 and response.get_json()["code"] == "model_unavailable"


def test_version_mismatch_is_not_silently_ignored(monkeypatch):
    import ml.inference
    original_version = importlib.metadata.version
    monkeypatch.setattr(ml.inference.importlib.metadata, "version", lambda name: "9.9.9" if name == "scikit-learn" else original_version(name))
    monkeypatch.setattr(Config, "MODEL_BACKEND", "calibrated")
    service = NewsAnalyzerService()
    assert not service.is_model_loaded()
    assert "scikit-learn diverge" in service.model_error


def test_short_text_adjustment_is_preserved(service):
    data = service.analyze({"title": "", "text": "pesquisa oficial"})
    expected = service.calibrated.motor.prever("", "pesquisa oficial")
    assert data["ml_details"]["fator_texto_curto"] == pytest.approx(2 / 33)
    assert data["prob_fake"] == expected["prob_fake"]
    assert data["faixa"] == "inconclusiva"


def test_inference_failure_returns_503(client, service, monkeypatch):
    monkeypatch.setattr("backend.routes.scraper_service.scrape", lambda url: {"text": BODY})
    def fail(*args):
        raise ValueError("falha de inferência")
    monkeypatch.setattr(service.calibrated.motor, "prever", fail)
    assert client.post("/api/analyze", json={"url": URL}).status_code == 503


@pytest.mark.parametrize("payload", [None, [], "not-json-object", 42])
def test_api_rejects_non_object_json(client, payload):
    assert client.post("/api/analyze", data=json.dumps(payload), content_type="application/json").status_code == 400


def test_scraping_failure_does_not_return_a_prediction(client, monkeypatch):
    def fail(url):
        raise ValueError("Não foi possível extrair a notícia.")
    monkeypatch.setattr("backend.routes.scraper_service.scrape", fail)
    response = client.post("/api/analyze", json={"url": URL})
    assert response.status_code == 422
    assert "confidence_score" not in response.get_json()


def test_api_documentation_and_contract(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/api").get_json()["docs"] == "/docs"
    schema = client.get("/api/openapi.json").get_json()
    assert schema["openapi"] == "3.0.3"
    assert "503" in schema["paths"]["/api/analyze"]["post"]["responses"]


def test_malformed_json_returns_json_error(client):
    response = client.post("/api/analyze", data='{"url":', content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"
