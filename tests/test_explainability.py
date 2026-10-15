import importlib.metadata
import json

import numpy as np
import pytest

from backend.services.explainer import explain_prediction, sentence_spans

TITLE = "Boletim sobre um assunto público"
BODY = (
    "Compartilhe esta mensagem antes que apaguem. "
    "O relatório apresenta dados com metodologia. "
    "Os demais detalhes foram divulgados ontem."
)


class Predictor:
    def __init__(self):
        self.documents = []

    def score(self, title, body):
        return 0.5 + 0.25 * ("antes que apaguem" in body) - 0.2 * ("com metodologia" in body)

    def predict_documents_proba(self, documents):
        self.documents.extend(documents)
        return [[self.score(t, b), 1 - self.score(t, b)] for t, b in documents]


def prediction(predictor, title=TITLE, body=BODY):
    return {"prob_fake": predictor.score(title, body), "faixa": "inconclusiva"}


def test_effects_use_final_probabilities_in_both_directions_and_exact_original_spans():
    predictor = Predictor()
    original = prediction(predictor)
    result = explain_prediction(predictor, TITLE, BODY, original)
    assert result["scope"] == "final_prediction" and result["status"] == "ok"
    assert len(predictor.documents) == 4
    suspect, reliable = result["trechos"]
    assert suspect["efeito_pp"] == pytest.approx(25)
    assert reliable["efeito_pp"] == pytest.approx(-20)
    assert suspect["direcao"] == "favorece_falsa"
    assert reliable["direcao"] == "favorece_confiavel"
    for item in result["avaliacoes"]:
        source = TITLE if item["origem"] == "titulo" else BODY
        assert source[item["inicio"]:item["fim"]] == item["texto"]
        changed = ("", BODY) if item["origem"] == "titulo" else (TITLE, BODY[:item["inicio"]] + BODY[item["fim"]:])
        assert item["prob_fake_sem_trecho"] == predictor.score(*changed)
    assert "30,0%" in result["resumo"] and "75,0%" in result["resumo"]
    assert original == prediction(predictor)
    json.dumps(result, allow_nan=False)


def test_sentence_offsets_preserve_abbreviations_decimals_quotes_and_newlines():
    body = '  A Dra. Ana publicou 2.5 pontos.\n“Os dados não confirmam essa afirmação!” Outra frase permanece sem ponto'
    spans = [body[start:end] for start, end in sentence_spans(body)]
    assert spans == ["A Dra. Ana publicou 2.5 pontos.", '“Os dados não confirmam essa afirmação!”', "Outra frase permanece sem ponto"]


@pytest.mark.parametrize("budget,expected", [(1, 1), (8, 8), (1000, 16)])
def test_inference_budget_is_bounded_and_sampling_is_reported(budget, expected):
    body = " ".join(f"A frase número {i} contém uma informação." for i in range(40))
    predictor = Predictor()
    result = explain_prediction(predictor, TITLE, body, prediction(predictor, TITLE, body), budget)
    assert len(predictor.documents) == expected
    assert result["cobertura"]["trechos_avaliados"] == expected
    assert result["cobertura"]["trechos_identificados"] == 41
    assert result["cobertura"]["parcial"]
    if expected > 2:
        assert "número 39" in result["avaliacoes"][-1]["texto"]


def test_constant_model_does_not_invent_reasons_for_suspicious_language():
    predictor = Predictor()
    predictor.score = lambda title, body: 0.9
    result = explain_prediction(predictor, TITLE, BODY, {"prob_fake": 0.9, "faixa": "falsa"})
    assert result["status"] == "no_effect" and not result["trechos"]
    assert "influência expressiva" in result["resumo"]
    assert "Compartilhe" not in result["resumo"]


def test_insufficient_text_does_not_score_an_empty_perturbation():
    predictor = Predictor()
    result = explain_prediction(predictor, "", "Uma única frase sem título.", {"prob_fake": 0.5, "faixa": "inconclusiva"})
    assert result["status"] == "insufficient_text"
    assert not predictor.documents


def test_truncation_warning_does_not_claim_transformer_saw_entire_article():
    predictor = Predictor()
    original = {**prediction(predictor), "transformer": {"truncated": True, "max_length": 256, "used_tokens": 256, "input_tokens": 700}}
    result = explain_prediction(predictor, TITLE, BODY, original)
    assert result["cobertura"]["transformer_truncado"]
    assert result["cobertura"]["transformer_tokens_entrada"] == 700
    assert any("outras partes" in text and "256" in text for text in result["limitacoes"])


@pytest.mark.parametrize("scores", [[], [[float("nan"), 0]], [[1.2, -0.2]] * 4])
def test_invalid_perturbation_results_are_rejected(scores):
    predictor = Predictor()
    predictor.predict_documents_proba = lambda documents: scores
    with pytest.raises(ValueError):
        explain_prediction(predictor, TITLE, BODY, prediction(predictor))


@pytest.mark.skipif(importlib.metadata.version("scikit-learn") != "1.8.0", reason="O artefato importado exige sklearn 1.8.0")
def test_real_calibrated_batch_matches_original_motor_with_title_and_body():
    from backend.ml.calibrated import CalibratedClassifier
    from pathlib import Path
    model = CalibratedClassifier(Path(__file__).resolve().parents[1] / "models/classificador.joblib")
    documents = [(TITLE, BODY), ("<script>conteúdo ignorado", BODY), (TITLE, "Texto curto.")]
    probabilities = model.predict_documents_proba(documents)
    for document, row in zip(documents, probabilities):
        assert row[0] == pytest.approx(model.motor.prever(*document)["prob_fake"], abs=1e-12)
        assert sum(row) == pytest.approx(1)
