import pytest
from unittest.mock import patch
from app import create_app
from app.config import Config

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Testa o endpoint de status e health check."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "online"
    assert "criteria" in data
    assert len(data["criteria"]) == 5

def test_frontend_served_by_flask(client):
    """Testa se a raiz serve o index.html do VerificAI e os assets estáticos."""
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert b"VerificAI" in res_index.data

    res_css = client.get("/style.css")
    assert res_css.status_code == 200

    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    assert b"analyzeNews" in res_js.data

def test_analyze_missing_url(client):
    """Testa requisição sem URL."""
    response = client.post("/api/analyze", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"

def test_analyze_invalid_url(client):
    """Testa requisição com URL malformada."""
    response = client.post("/api/analyze", json={"url": "htp:/invalid-url"})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"

def test_analyze_localhost_forbidden(client):
    """Testa proteção SSRF contra localhost."""
    response = client.post("/api/analyze", json={"url": "http://localhost:8080/fake"})
    assert response.status_code == 400
    data = response.get_json()
    assert "localhost" in data["message"].lower()

@patch("app.routes.scraper_service.scrape")
def test_analyze_reliable_news(mock_scrape, client):
    """Testa fluxo de análise para uma notícia com características confiáveis (>= 50)."""
    mock_scrape.return_value = {
        "url": "https://g1.globo.com/ciencia/noticia-exemplo",
        "domain": "g1.globo.com",
        "title": "Pesquisa da USP revela avanços em tratamento com 85% de eficácia",
        "text": (
            "Uma nova pesquisa conduzida por especialistas da Universidade de São Paulo (USP) "
            "apresentou resultados promissores nesta semana. Segundo dados do relatório científico, "
            "o novo protocolo atingiu 85% de eficácia nos testes clínicos com 450 pacientes voluntários. "
            "\"Este é um marco fundamental para a ciência brasileira e reforça o investimento público\", "
            "declarou a coordenadora do estudo, Dra. Silva. Os dados completos foram submetidos a parecer "
            "dos pares em periódico internacional indexado."
        ),
        "author": "Mariana Silva",
        "publish_date": "2026-10-01",
        "word_count": 80,
        "external_links_count": 4
    }

    response = client.post("/api/analyze", json={"url": "https://g1.globo.com/ciencia/noticia-exemplo"})
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "success"
    assert 0 <= data["confidence_score"] <= 100
    assert data["confidence_score"] >= 50.0
    assert data["verdict"] == "confiavel"
    assert data["verdict_label"] == "Confiável"

    # Verificação dos 5 critérios
    criteria = data["criteria"]
    assert "evidencias" in criteria
    assert "qualidade_fontes" in criteria
    assert "corroboracao" in criteria
    assert "contexto" in criteria
    assert "atualidade" in criteria

    for crit in ["evidencias", "qualidade_fontes", "corroboracao", "contexto", "atualidade"]:
        assert "score" in criteria[crit]
        assert "label" in criteria[crit]
        assert "details" in criteria[crit]
        assert 0 <= criteria[crit]["score"] <= 100

@patch("app.routes.scraper_service.scrape")
def test_analyze_fake_news(mock_scrape, client):
    """Testa fluxo de análise para notícia alarmista / sensacionalista (< 50)."""
    mock_scrape.return_value = {
        "url": "http://blog-falso-urgente.xyz/bomba",
        "domain": "blog-falso-urgente.xyz",
        "title": "URGENTE! BOMBA! A MÍDIA ESCONDE O QUE DESCOBRIRAM!",
        "text": (
            "Urgente! Veja antes que apaguem! O que ninguém te conta sobre o milagre secreto que a grande mídia esconde! "
            "Compartilhe antes que saia do ar! É um absurdo inacreditável! Uma conspiração mundial desmascarada!"
        ),
        "author": "admin",
        "publish_date": None,
        "word_count": 35,
        "external_links_count": 0
    }

    response = client.post("/api/analyze", json={"url": "http://blog-falso-urgente.xyz/bomba"})
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "success"
    assert data["confidence_score"] < 50.0
    assert data["verdict"] == "desconfiavel"
    assert "Não Confiável" in data["verdict_label"]

def test_real_linearsvc_model_prediction(client):
    """Testa se o modelo LinearSVC real em produção infere corretamente e retorna ml_details."""
    # Garante reload do modelo real
    reload_res = client.post("/api/reload-model")
    assert reload_res.status_code == 200
    assert reload_res.get_json()["custom_ml_model_loaded"] is True

    # 1. Teste de notícia real / confiável com vocabulário factual
    with patch("app.routes.scraper_service.scrape") as mock_scrape:
        mock_scrape.return_value = {
            "url": "https://g1.globo.com/saude/pesquisa",
            "domain": "g1.globo.com",
            "title": "Ministério da Saúde divulga dados sobre vacinação e relatórios",
            "text": (
                "O Ministério da Saúde informou nesta quinta-feira que o índice de imunização "
                "atingiu a meta nacional estabelecida pelo Programa Nacional de Imunizações (PNI). "
                "Segundo dados oficiais e parecer de pesquisadores da Fiocruz e da Anvisa, "
                "foram distribuídas milhões de doses conforme cronograma estabelecido."
            ),
            "author": "Redação G1",
            "publish_date": "2026-10-01",
            "word_count": 50,
            "external_links_count": 3
        }
        res = client.post("/api/analyze", json={"url": "https://g1.globo.com/saude/pesquisa"})
        assert res.status_code == 200
        data = res.get_json()

        assert data["used_custom_ml_model"] is True
        assert "ml_details" in data
        assert data["ml_details"]["model_type"] == "LinearSVC (FakeRecogna)"
        assert "hyperplane_distance" in data["ml_details"]
        assert data["confidence_score"] >= 50.0
        assert data["verdict"] == "confiavel"

    # 2. Teste de fake news com vocabulário típico da base FakeRecogna
    with patch("app.routes.scraper_service.scrape") as mock_scrape:
        mock_scrape.return_value = {
            "url": "http://site-suspeito.xyz/bomba",
            "domain": "site-suspeito.xyz",
            "title": "Boato: Papa Francisco foi preso no Vaticano sob acusação de fraude e tráfico",
            "text": (
                "Boato: Papa Francisco foi preso no Vaticano sob acusação de fraude e tráfico humano "
                "em operação da polícia militar vazada na internet. Compartilhe antes que saia do ar! "
                "É um absurdo inacreditável! Uma conspiração mundial desmascarada!"
            ),
            "author": "admin",
            "publish_date": None,
            "word_count": 35,
            "external_links_count": 0
        }
        res = client.post("/api/analyze", json={"url": "http://site-suspeito.xyz/bomba"})
        assert res.status_code == 200
        data = res.get_json()

        assert data["used_custom_ml_model"] is True
        assert data["ml_details"]["predicted_class"] == 1
        assert data["confidence_score"] < 50.0
        assert data["verdict"] == "desconfiavel"


