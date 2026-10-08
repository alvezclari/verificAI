# Modelos de Machine Learning (Pasta `models/`)

Esta pasta é destinada aos arquivos do modelo de Machine Learning que você treinar.

## Como plugar seu modelo treinado:

1. Salve seu modelo e vetorizador em formato pickle usando `joblib` ou `pickle`:
   ```python
   import joblib

   # Se você usar um Pipeline do Scikit-Learn (Vetorizador + Classificador juntos):
   joblib.dump(meu_pipeline, "models/model.pkl")

   # Ou se salvar separadamente:
   joblib.dump(meu_modelo, "models/model.pkl")
   joblib.dump(meu_vectorizer, "models/vectorizer.pkl")
   ```

2. Cole os arquivos nesta pasta (`models/`):
   - `model.pkl` (obrigatório para ativar o modelo de ML próprio)
   - `vectorizer.pkl` (opcional, caso não faça parte de um Pipeline único)

3. O backend em `app/services/analyzer.py` detectará automaticamente os arquivos ao inicializar ou processar as requisições.

4. Você pode checar se o modelo foi detectado acessando o endpoint:
   `GET http://localhost:5000/api/health`
   Que retornará `"model_loaded": true`.

Enquanto nenhum arquivo `.pkl` for colocado nesta pasta, o backend usará um analisador baseline (heurístico/NLP) totalmente funcional para não bloquear os testes do frontend!

