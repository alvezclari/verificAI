import logging
from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest
from backend.utils.validators import validate_url
from backend.services.scraper import ScraperService
from backend.services.analyzer import NewsAnalyzerService
from backend.ml.errors import ModelUnavailableError

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Instanciação dos serviços
scraper_service = ScraperService()
analyzer_service = NewsAnalyzerService()

@api_bp.route("", methods=["GET"])
@api_bp.route("/", methods=["GET"])
def api_index():
    return jsonify({"name": "VerificAI API", "docs": "/docs", "openapi": "/api/openapi.json", "health": "/api/health"})


@api_bp.route("/openapi.json", methods=["GET"])
def openapi_schema():
    from backend.openapi import OPENAPI
    return jsonify(OPENAPI)

@api_bp.route("/health", methods=["GET"])
def health_check():
    """
    Verifica o estado da API e se o modelo de ML do usuário foi carregado.
    """
    return jsonify({
        "status": "online" if analyzer_service.is_model_loaded() else "degraded",
        "model": analyzer_service.model_status(),
        "custom_ml_model_loaded": analyzer_service.is_model_loaded(),
        "ensemble_loaded": analyzer_service.ensemble is not None,
        "model_error": analyzer_service.model_error,
        "threshold": None if analyzer_service.probability_thresholds() else 50.0,
        "thresholds": analyzer_service.probability_thresholds(),
        "criteria": [
            "Evidencias",
            "Qualidade das fontes",
            "Corroboracao",
            "Contexto",
            "Atualidade"
        ]
    }), 200

@api_bp.route("/reload-model", methods=["POST"])
def reload_model():
    """
    Recarrega o modelo de ML da pasta models/ sem precisar reiniciar o backend.
    """
    loaded = analyzer_service.reload_model()
    return jsonify({
        "status": "success" if loaded else "error",
        "model": analyzer_service.model_status(),
        "custom_ml_model_loaded": loaded,
        "ensemble_loaded": analyzer_service.ensemble is not None,
        "model_error": analyzer_service.model_error,
        "message": "Modelo carregado com sucesso!" if loaded else "Modelo indisponível; consulte model_error e os artefatos em models/."
    }), 200 if loaded else 503

@api_bp.route("/analyze", methods=["POST"])
def analyze_news():
    """
    Endpoint principal consumido pelo frontend.
    Recebe { "url": "https://..." }
    Executa o web scraping e em seguida a avaliação de confiabilidade e os 5 critérios.
    """
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "O corpo da requisição deve ser um JSON válido com o campo 'url'."
        }), 400

    try:
        data = request.get_json()
    except BadRequest:
        return jsonify({"status": "error", "message": "O corpo deve conter um JSON válido."}), 400
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Envie um objeto JSON com o campo 'url'."}), 400
    url = data.get("url")

    # 1. Validação da URL
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 400

    if analyzer_service.backend != "legacy" and not analyzer_service.is_model_loaded():
        return jsonify({
            "status": "error", "code": "model_unavailable",
            "message": "Modelo indisponível. Consulte /api/health e as dependências do classificador.",
        }), 503

    # 2. Web Scraping
    try:
        article_data = scraper_service.scrape(url)
    except ValueError as ve:
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 422
    except Exception as e:
        logger.exception(f"Erro inesperado no web scraping: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erro interno ao processar a página: {str(e)}"
        }), 500

    # 3. Análise pelo Modelo de ML e Cálculo dos 5 Critérios
    try:
        analysis_result = analyzer_service.analyze(article_data)
        return jsonify(analysis_result), 200
    except ModelUnavailableError:
        logger.exception("Modelo indisponível durante a análise")
        return jsonify({
            "status": "error", "code": "model_unavailable",
            "message": "O modelo não conseguiu analisar o texto extraído. Consulte /api/health.",
        }), 503
    except Exception as e:
        logger.exception(f"Erro na análise de ML: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erro interno ao processar a análise da notícia: {str(e)}"
        }), 500

