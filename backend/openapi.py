"""Contrato HTTP da API local."""

OPENAPI = {
    "openapi": "3.0.3",
    "info": {"title": "VerificAI API", "version": "1.1.0", "description": "Classificação textual calibrada de notícias por URL."},
    "servers": [{"url": "/"}],
    "paths": {
        "/api/health": {"get": {"summary": "Estado e versão do modelo", "responses": {
            "200": {"description": "Estado do serviço; consulte model.loaded e model.error."},
        }}},
        "/api/analyze": {"post": {
            "summary": "Extrair uma notícia e classificá-la",
            "requestBody": {"required": True, "content": {"application/json": {"schema": {
                "type": "object", "required": ["url"],
                "properties": {"url": {"type": "string", "format": "uri", "example": "https://exemplo.com.br/noticia"}},
            }}}},
            "responses": {
                "200": {"description": "Probabilidade calibrada, faixa, métricas do artefato, explicações e sinais auxiliares.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Analysis"}}}},
                "400": {"description": "JSON ou URL inválidos."},
                "422": {"description": "Página inacessível ou sem texto extraível."},
                "503": {"description": "Modelo indisponível ou falha de inferência; não há score substituto."},
                "500": {"description": "Falha interna inesperada."},
            },
        }},
        "/api/reload-model": {"post": {"summary": "Recarregar o modelo selecionado", "responses": {
            "200": {"description": "Modelo recarregado."}, "503": {"description": "Falha ao carregar o modelo."},
        }}},
    },
    "components": {"schemas": {"Analysis": {
        "type": "object", "required": ["status", "confidence_score", "verdict", "ml_details"],
        "properties": {
            "status": {"type": "string", "enum": ["success"]},
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 100,
                                 "description": "No modo calibrated, 100 × (1 − prob_fake)."},
            "prob_fake": {"type": "number", "minimum": 0, "maximum": 1, "nullable": True},
            "faixa": {"type": "string", "enum": ["verdadeira", "inconclusiva", "falsa"], "nullable": True},
            "verdict": {"type": "string", "enum": ["confiavel", "inconclusiva", "desconfiavel"]},
            "verdict_label": {"type": "string"}, "model_version": {"type": "string", "nullable": True},
            "thresholds": {"type": "object", "nullable": True}, "ml_details": {"type": "object", "nullable": True},
            "model_evaluation": {"type": "object", "nullable": True},
            "criteria": {"type": "object"}, "criteria_affect_score": {"type": "boolean"},
            "provenance": {"type": "object"},
        },
    }}},
}
