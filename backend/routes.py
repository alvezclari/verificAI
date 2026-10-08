import logging
from flask import Blueprint, request, jsonify
from app.utils.validators import validate_url
from app.services.scraper import ScraperService
from app.services.analyzer import NewsAnalyzerService

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Instanciação dos serviços
scraper_service = ScraperService()
analyzer_service = NewsAnalyzerService()

@api_bp.route("/health", methods=["GET"])
def health_check():
    """
    Verifica o estado da API e se o modelo de ML do usuário foi carregado.
    """
    return jsonify({
        "status": "online",
        "custom_ml_model_loaded": analyzer_service.is_model_loaded(),
        "threshold": 50.0,
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
        "status": "success",
        "custom_ml_model_loaded": loaded,
        "message": "Modelo carregado com sucesso!" if loaded else "Nenhum arquivo model.pkl encontrado em models/."
    }), 200

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

    data = request.get_json()
    url = data.get("url")

    # 1. Validação da URL
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 400

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
    except Exception as e:
        logger.exception(f"Erro na análise de ML: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erro interno ao processar a análise da notícia: {str(e)}"
        }), 500

