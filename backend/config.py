import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-fake-news-detector")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1")
    
    # ML Models paths
    MODELS_DIR = MODELS_DIR
    MODEL_PATH = MODELS_DIR / "model.pkl"
    VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
    
    @classmethod
    def get_model_path(cls):
        """Retorna o primeiro arquivo de modelo válido encontrado na pasta models."""
        possible_names = [
            "best_linearsvc_model.joblib",
            "model.pkl",
            "model.joblib",
            "modelo_linearsvc_fakerecogna.joblib",
            "linearsvc_campeao.pkl"
        ]
        for name in possible_names:
            p = MODELS_DIR / name
            if p.exists():
                return p
        return cls.MODEL_PATH
    
    # Scraper settings
    SCRAPER_TIMEOUT = int(os.getenv("SCRAPER_TIMEOUT", "15"))  # seconds
    SCRAPER_USER_AGENT = os.getenv(
        "SCRAPER_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    
    # Threshold de confiabilidade (0 a 100)
    CONFIDENCE_THRESHOLD = 50.0

