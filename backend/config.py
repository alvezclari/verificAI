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
    # Ativa o ensemble supervisionado somente após a publicação de um bundle completo.
    DEEP_ENSEMBLE_MANIFEST = MODELS_DIR / "mdeberta-ensemble" / "manifest.json"
    MODEL_BACKEND = os.getenv("MODEL_BACKEND", "ensemble" if DEEP_ENSEMBLE_MANIFEST.exists() else "calibrated")
    CLASSIFIER_PATH = Path(os.getenv("CLASSIFIER_PATH", str(MODELS_DIR / "classificador.joblib")))
    VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
    ENSEMBLE_MANIFEST = Path(os.getenv("ENSEMBLE_MANIFEST", str(DEEP_ENSEMBLE_MANIFEST if DEEP_ENSEMBLE_MANIFEST.exists() else MODELS_DIR / "ensemble" / "manifest.json")))
    ENSEMBLE_ENABLED = os.getenv("ENSEMBLE_ENABLED", "true").lower() in ("true", "1")
    TRANSFORMER_DEVICE = os.getenv("TRANSFORMER_DEVICE", "auto")
    XAI_ENABLED = os.getenv("XAI_ENABLED", "true").lower() in ("true", "1")
    XAI_MAX_EVALUATIONS = int(os.getenv("XAI_MAX_EVALUATIONS", "8"))
    # Convenções dos artefatos legados: SVC 0=real; exemplo LogReg 1=real.
    PROBABILISTIC_RELIABLE_CLASS = int(os.getenv("PROBABILISTIC_RELIABLE_CLASS", "1"))
    
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

