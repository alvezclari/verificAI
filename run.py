import os
from backend import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1")
    print(f"[*] Iniciando servidor Fake News Detector Backend na porta {port}...")
    print(f"[*] Acesse http://localhost:{port}/ ou teste a rota /api/health")
    app.run(host="0.0.0.0", port=port, debug=debug)

