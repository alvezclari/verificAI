import os
import logging
from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from backend.config import Config
from backend.routes import api_bp

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "verificAI-front_inicial"

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configuração de CORS para permitir requisições de qualquer origem
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Configuração de Logs
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )

    # Registro das rotas da API
    app.register_blueprint(api_bp)

    # Servir o Frontend VerificAI na raiz e arquivos estáticos (CSS, JS, imagens)
    @app.route("/", methods=["GET"])
    def index():
        if (FRONTEND_DIR / "index.html").exists():
            return send_from_directory(FRONTEND_DIR, "index.html")
        return jsonify({
            "name": "Backend Fake News Detector & News Reliability API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/api/health",
                "analyze": "/api/analyze [POST]",
                "reload_model": "/api/reload-model [POST]"
            }
        })

    @app.route("/<path:filename>", methods=["GET"])
    def serve_frontend_assets(filename):
        # Permite servir style.css, app.js e qualquer asset do front
        if (FRONTEND_DIR / filename).exists():
            return send_from_directory(FRONTEND_DIR, filename)
        return jsonify({"error": "Arquivo não encontrado"}), 404

    return app
