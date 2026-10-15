"""Contrato HTTP da API local."""

OPENAPI = {
    "openapi": "3.0.3",
    "info": {"title": "VerificAI API", "version": "1.2.0", "description": "Classificação textual calibrada de notícias por URL, com explicação local por trechos."},
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
            "thresholds": {"type": "object", "nullable": True},
            "ml_details": {"type": "object", "nullable": True, "properties": {
                "explicacao": {"type": "object", "properties": {
                    "local": {"$ref": "#/components/schemas/LocalExplanation"},
                }},
            }},
            "model_evaluation": {"type": "object", "nullable": True},
            "criteria": {"type": "object"}, "criteria_affect_score": {"type": "boolean"},
            "provenance": {"type": "object"},
        },
    }}},
}

OPENAPI["components"]["schemas"].update({
    "ExplanationSpan": {
        "type": "object",
        "required": ["origem", "inicio", "fim", "texto", "prob_fake_sem_trecho", "efeito_pp", "direcao"],
        "properties": {
            "origem": {"type": "string", "enum": ["titulo", "corpo"]},
            "inicio": {"type": "integer", "minimum": 0, "description": "Offset em caracteres Unicode no título ou corpo extraído."},
            "fim": {"type": "integer", "minimum": 0, "description": "Fim exclusivo do trecho no mesmo campo."},
            "texto": {"type": "string"},
            "prob_fake_sem_trecho": {"type": "number", "minimum": 0, "maximum": 1},
            "efeito_pp": {"type": "number", "description": "100 × (P(falsa) original − P(falsa) sem o trecho). Positivo favorece a classe falsa."},
            "direcao": {"type": "string", "enum": ["favorece_falsa", "favorece_confiavel", "efeito_pequeno"]},
        },
    },
    "LocalExplanation": {
        "type": "object", "description": "Comparação local por remoção de trechos sobre a probabilidade final, sem checagem factual.",
        "required": ["metodo", "scope", "status", "resumo", "trechos", "avaliacoes", "limitacoes"],
        "properties": {
            "metodo": {"type": "string", "enum": ["sentence_deletion"]},
            "scope": {"type": "string", "enum": ["final_prediction"]},
            "status": {"type": "string", "enum": ["ok", "no_effect", "insufficient_text", "unavailable"]},
            "resumo": {"type": "string"},
            "prob_fake_original": {"type": "number", "minimum": 0, "maximum": 1},
            "limiar_efeito_pp": {"type": "number"},
            "trechos": {"type": "array", "maxItems": 6, "items": {"$ref": "#/components/schemas/ExplanationSpan"}},
            "avaliacoes": {"type": "array", "maxItems": 16, "items": {"$ref": "#/components/schemas/ExplanationSpan"}},
            "cobertura": {"type": "object", "description": "Contagens, amostragem parcial e cobertura do ramo semântico."},
            "limitacoes": {"type": "array", "items": {"type": "string"}},
            "tempo_ms": {"type": "number", "minimum": 0},
        },
    },
})
