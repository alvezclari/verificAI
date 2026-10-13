import json
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from backend.config import Config
from backend.ml.data import load_records, split_records
from backend.ml.ensemble import LexicalClassifier, SoftVotingEnsemble, softmax
from backend.ml.training import build_lexical_model
from backend.services.analyzer import NewsAnalyzerService


def test_aligns_reversed_classes_and_requires_probabilities():
    model = Mock(classes_=np.array([1, 0]))
    model.predict_proba.return_value = np.array([[0.8, 0.2]])
    assert LexicalClassifier(model).predict_proba(["texto"])[0].tolist() == [0.2, 0.8]
    with pytest.raises(ValueError, match="predict_proba"):
        LexicalClassifier(object())


def test_parallel_raw_text_weighted_voting():
    barrier = threading.Barrier(2)
    text = "URGENTE: São Paulo não confirmou a alegação!"

    def component(probabilities):
        class Predictor:
            def predict_proba(self, texts):
                assert texts == [text]
                barrier.wait(timeout=5)
                return np.array([probabilities])
        return Predictor()

    result = SoftVotingEnsemble(component([0.8, 0.2]), component([0.1, 0.9]), weights=[1, 3]).predict(text)
    assert result["probabilities"]["confiavel"] == pytest.approx(0.725)
    assert result["predicted_class"] == 1
    assert result["weights"] == {"lexical": 0.25, "transformer": 0.75}


@pytest.mark.parametrize("weights", [[0, 1], [-1, 1], [float("nan"), 1], [1], [1, float("inf")]])
def test_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        SoftVotingEnsemble(None, None, weights)


def test_softmax_numerically_stable_and_invalid_output_rejected():
    assert softmax([[10000, 10000]]).tolist() == [[0.5, 0.5]]
    with pytest.raises(ValueError):
        softmax([[float("nan"), 1]])
    invalid = Mock()
    invalid.predict_proba.return_value = [[0.9, 0.9]]
    with pytest.raises(ValueError, match="Probabilidades"):
        SoftVotingEnsemble(invalid, invalid).predict("texto")


def test_logreg_uses_predict_proba_before_decision_function(monkeypatch):
    monkeypatch.setattr(Config, "MODEL_BACKEND", "legacy")
    monkeypatch.setattr(Config, "ENSEMBLE_ENABLED", False)
    model = build_lexical_model()
    model.fit(["boato mentira fraude", "boato mentira", "ciência pesquisa dados", "ciência dados oficiais"], [0, 0, 1, 1])
    analyzer = NewsAnalyzerService()
    analyzer.model, analyzer.vectorizer = model, None
    score, details = analyzer._predict_with_user_model("", "ciência dados oficiais")
    assert score == pytest.approx(model.predict_proba(["ciência dados oficiais"])[0, 1] * 100)
    assert details["predicted_class"] == 1


def test_ensemble_verdict_does_not_mix_in_heuristics(monkeypatch):
    monkeypatch.setattr(Config, "MODEL_BACKEND", "legacy")
    monkeypatch.setattr(Config, "ENSEMBLE_ENABLED", False)
    analyzer = NewsAnalyzerService()
    analyzer.model, analyzer.vectorizer = None, None
    analyzer.model_loaded = True
    predictor = Mock()
    predictor.predict_proba.return_value = [[0.49, 0.51]]
    analyzer.ensemble = SoftVotingEnsemble(predictor, predictor)
    result = analyzer.analyze({"title": "BOMBA!!! URGENTE!!!", "text": "Compartilhe antes que apaguem!"})
    assert result["confidence_score"] == pytest.approx(51)
    assert result["verdict"] == "confiavel"
    assert len(result["criteria"]) == 5
    predictor.predict_proba.side_effect = RuntimeError("indisponível")
    with pytest.raises(RuntimeError, match="inferência"):
        analyzer.analyze({"text": "texto"})


def test_invalid_manifest_is_visible_and_reload_clears_stale_state(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{}')
    monkeypatch.setattr(Config, "MODEL_BACKEND", "ensemble")
    monkeypatch.setattr(Config, "ENSEMBLE_ENABLED", True)
    monkeypatch.setattr(Config, "ENSEMBLE_MANIFEST", manifest)
    analyzer = NewsAnalyzerService()
    assert analyzer.model_error
    assert not analyzer.is_model_loaded()
    with pytest.raises(RuntimeError, match="indisponível"):
        analyzer.analyze({"text": "texto"})
    monkeypatch.setattr(Config, "MODEL_BACKEND", "legacy")
    monkeypatch.setattr(Config, "ENSEMBLE_ENABLED", False)
    monkeypatch.setattr(Config, "get_model_path", lambda: tmp_path / "absent.pkl")
    assert not analyzer.reload_model()
    assert analyzer.model is None and analyzer.ensemble is None and analyzer.model_error is None


def test_manifest_paths_cannot_escape_bundle(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "labels": {"0": "fake", "1": "confiavel"},
                                "weights": [1, 1], "lexical_path": "../outside.joblib"}))
    with pytest.raises(ValueError, match="diretório"):
        SoftVotingEnsemble.from_manifest(path)


def test_raw_csv_labels_deduplication_and_split_reproducibility(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text('text,label\n"Não há confirmação!",real\n" não   há confirmação! ",real\n"Milagre secreto",fake\n')
    texts, labels, source = load_records(path, "text", "label", "real")
    assert texts == ["Não há confirmação!", "Milagre secreto"]
    assert labels == [1, 0] and source["duplicates_removed"] == 1
    path.write_text('text,label\ntexto,real\ntexto,fake\n')
    with pytest.raises(ValueError, match="conflitantes"):
        load_records(path, "text", "label", "real")
    texts, labels = [f"notícia {i}" for i in range(40)], [i % 2 for i in range(40)]
    splits, metadata = split_records(texts, labels)
    assert (splits, metadata) == split_records(texts, labels)
    train, val, test = [set(splits[k]) for k in ("train", "validation", "test")]
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(range(40))
