import importlib.metadata
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from backend.ml.deep_ensemble import CalibratedDeepEnsemble, calibrated_blend
from backend.ml.deep_training import fit_temperature, grouped_splits
from backend.ml.deep_calibration import fit_bias_calibration, select_combination


def test_blend_alignment_weights_and_calibration():
    lexical = [[0.8, 0.2]]
    semantic = [[0.1, 0.9]]
    np.testing.assert_allclose(calibrated_blend(lexical, semantic, [1, 3]), [[0.275, 0.725]])
    result = calibrated_blend(lexical, semantic, [1, 3], temperature=2)
    assert 0.5 < result[0, 1] < 0.725
    with pytest.raises(ValueError):
        calibrated_blend(lexical, semantic, [1, 1], temperature=0)


def test_grouped_split_is_disjoint_including_aligned_pairs():
    labels = [label for _ in range(80) for label in [0, 1]]
    groups = [str(i) for i in range(80) for _ in range(2)]
    splits = grouped_splits(labels, groups)
    assert splits == grouped_splits(labels, groups)
    assert set(sum(splits.values(), [])) == set(range(160))
    for name, indices in splits.items():
        assert {labels[i] for i in indices} == {0, 1}
        for other, others in splits.items():
            if name != other:
                assert not {groups[i] for i in indices} & {groups[i] for i in others}


def test_temperature_fitted_without_using_test_data():
    logits = [[-8, 8], [-8, 8], [8, -8], [8, -8]]
    labels = [1, 0, 0, 1]
    assert fit_temperature(logits, labels) > 1


def test_bias_calibration_corrects_imported_branch_prior_shift():
    # Ambos os ramos preservam sua saída; a combinação aprende um intercepto na reserva.
    probabilities = np.array([[0.4, 0.6], [0.35, 0.65], [0.1, 0.9], [0.15, 0.85]])
    labels = np.array([0, 0, 1, 1])
    temperature, bias = fit_bias_calibration(probabilities, labels)
    final = calibrated_blend(probabilities, probabilities, [0.5, 0.5], temperature, bias)
    assert (final[:, 1] >= 0.5).tolist() == [False, False, True, True]
    chosen, candidates = select_combination(probabilities, probabilities, labels, probabilities, probabilities, labels)
    assert len(candidates) == 5 and chosen["validation_f1_macro"] == 1
    assert all(0 < w < 1 for w in chosen["weights"])


def test_components_affect_api_score_and_preserve_inconclusive_band(monkeypatch):
    from backend.config import Config
    from backend.services.analyzer import NewsAnalyzerService

    class Lexical:
        def predict_document(self, title, body):
            return {"prob_fake": 0.1, "fator_texto_curto": 1,
                    "explicacao": {"sustenta": [], "enfraquece": [], "limitacoes": []}}

    class Semantic:
        metadata = {"source_model": "mdeberta-test", "task": "supervised_fake_news"}
        max_length, device, temperature = 256, "cpu", 1

        def predict_proba(self, texts):
            assert texts == ["título\n\ncorpo"]
            return np.array([[0.9, 0.1]])

    ensemble = CalibratedDeepEnsemble(Lexical(), Semantic())
    result = ensemble.predict_document("título", "corpo")
    assert result["prob_fake"] == pytest.approx(0.5)
    assert result["faixa"] == "inconclusiva"
    assert result["component_probabilities"]["transformer"]["fake"] == 0.9
    monkeypatch.setattr(Config, "MODEL_BACKEND", "legacy")
    analyzer = NewsAnalyzerService()
    analyzer.model, analyzer.ensemble = None, ensemble
    analyzer.model_loaded = True
    analysis = analyzer.analyze({"title": "título", "text": "corpo"})
    assert analysis["verdict"] == "inconclusiva"
    assert analysis["score_method"] == "ensemble"
    assert analysis["confidence_score"] == pytest.approx(50)
    assert not analysis["criteria_affect_score"]
    original_predict = ensemble.lexical.predict_document
    def reload_during_prediction(title, body):
        analyzer.ensemble = None
        return original_predict(title, body)
    monkeypatch.setattr(ensemble.lexical, "predict_document", reload_during_prediction)
    # A requisição conserva o modelo capturado, mesmo se um reload trocar a referência.
    again = analyzer.analyze({"title": "título", "text": "corpo"})
    assert again["confidence_score"] == pytest.approx(50)
    assert again["score_method"] == "ensemble"


@pytest.mark.skipif(importlib.metadata.version("scikit-learn") != "1.8.0", reason="Artefato importado exige sklearn 1.8.0")
def test_real_imported_model_and_supervised_transformer_bundle(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from backend.ml.calibrated import CalibratedClassifier
    from backend.ml.ensemble import SoftVotingEnsemble
    from backend.ml.transformer import TransformerClassifier
    from backend.config import Config
    from backend.services.analyzer import NewsAnalyzerService

    torch.set_num_threads(1)
    checkpoint = tmp_path / "transformer"
    checkpoint.mkdir()
    vocab = checkpoint / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\npesquisa\noficial\nboato\nsecreto\n")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    model = transformers.BertForSequenceClassification(transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64, num_labels=2,
        id2label={0: "fake", 1: "confiavel"}, label2id={"fake": 0, "confiavel": 1},
    ))
    # Um passo supervisionado real exercita a cabeça binária usada no contrato.
    inputs = tokenizer(["boato secreto", "pesquisa oficial"], padding=True, return_tensors="pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(**inputs, labels=torch.tensor([0, 1])).loss.backward()
    optimizer.step()
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    metadata = {"schema_version": 1, "labels": {"0": "fake", "1": "confiavel"},
                "task": "supervised_fake_news", "max_length": 32, "temperature": 1.2,
                "split_signature": "test-signature", "source_model": "tiny-test-only"}
    (checkpoint / "training_metadata.json").write_text(json.dumps(metadata))
    classifier = Path(__file__).resolve().parents[1] / "models/classificador.joblib"
    shutil.copy2(classifier, tmp_path / "classificador.joblib")
    lexical = CalibratedClassifier(classifier)
    text = "pesquisa oficial"
    np.testing.assert_allclose(lexical.predict_proba([text])[0, 0], lexical.predict_document("", text)["prob_fake"])
    manifest = {
        "schema_version": 2, "labels": metadata["labels"], "version": "test-v2",
        "calibrated_path": "classificador.joblib", "calibrated_sha256": lexical.sha256,
        "transformer_path": "transformer", "split_signature": "test-signature",
        "weights": [0.5, 0.5], "temperature": 1.1,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    ensemble = SoftVotingEnsemble.from_manifest(path)
    result = ensemble.predict_document("", text)
    assert result["versao_modelo"] == "test-v2"
    assert result["transformer"]["task"] == "supervised_fake_news"
    assert result["transformer"]["input_tokens"] > 0
    assert not result["transformer"]["truncated"]
    assert result["explicacao"]["scope"] == "lexical_component_only"
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert np.isfinite(ensemble.predict_proba([text, "boato secreto"])).all()
    documents = [("", text), ("pesquisa", "boato secreto"), ("", "pesquisa oficial boato")]
    batched = ensemble.predict_documents_proba(documents)
    individual = [ensemble.predict_document(title, body)["prob_fake"] for title, body in documents]
    np.testing.assert_allclose(batched[:, 0], individual, rtol=1e-6, atol=1e-7)
    monkeypatch.setattr(Config, "MODEL_BACKEND", "ensemble")
    monkeypatch.setattr(Config, "ENSEMBLE_MANIFEST", path)
    monkeypatch.setattr(Config, "TRANSFORMER_DEVICE", "cpu")
    analyzer = NewsAnalyzerService()
    assert analyzer.model_status()["transformer"]["loaded"]
    from backend import routes, create_app
    monkeypatch.setattr(routes, "analyzer_service", analyzer)
    monkeypatch.setattr(routes.scraper_service, "scrape", lambda url: {"url": url, "title": "", "text": text})
    with create_app().test_client() as client:
        response = client.post("/api/analyze", json={"url": "https://example.com/noticia"})
        assert response.status_code == 200
        assert response.get_json()["model_version"] == "test-v2"
        assert response.get_json()["ml_details"]["component_probabilities"]["transformer"]
    metadata["task"] = "natural_language_inference"
    (checkpoint / "training_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="não foi ajustado"):
        CalibratedDeepEnsemble.from_manifest(path)
