import csv
import json

import numpy as np
import pytest

from backend.ml.training import main


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "news.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        for i in range(40):
            writer.writerow([f"{'pesquisa oficial ciência' if i % 2 else 'boato milagre secreto'} notícia {i}", i % 2])
    return path


def training_args(tmp_path, dataset):
    return ["--data", str(dataset), "--text-column", "text", "--label-column", "label", "--real-label", "1",
            "--output-dir", str(tmp_path / "bundle"), "--tracking-uri", f"sqlite:///{tmp_path / 'tracking.db'}"]


def test_lexical_training_records_parameters_metrics_and_artifacts(tmp_path, dataset, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    run_id = main([*training_args(tmp_path, dataset), "--stage", "lexical"])
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["lexical.clf__C"] == "1.0"
    assert run.data.params["lexical.tfidf__ngram_range"] == "(1, 2)"
    assert "lexical.test.f1_macro" in run.data.metrics
    assert "lexical.validation.log_loss" in run.data.metrics
    assert len(run.data.params["data.sha256"]) == 64
    assert (tmp_path / "bundle" / "lexical.joblib").exists()
    assert not (tmp_path / "bundle" / "manifest.json").exists()
    artifacts = mlflow.MlflowClient().list_artifacts(run_id, "bundle")
    assert {a.path for a in artifacts} == {"bundle/lexical.joblib", "bundle/lexical_metadata.json"}


def test_real_transformer_finetuning_tracking_and_local_ensemble(tmp_path, dataset, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from backend.ml.ensemble import SoftVotingEnsemble
    from backend.ml.transformer import TransformerClassifier

    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    base = tmp_path / "tiny-transformer"
    base.mkdir()
    vocab = base / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\npesquisa\noficial\nboato\nsecreto\nnoticia\n")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(base)
    model = transformers.BertForSequenceClassification(transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
        num_labels=2, id2label={0: "fake", 1: "confiavel"}, label2id={"fake": 0, "confiavel": 1},
    ))
    model.save_pretrained(base)
    args = [*training_args(tmp_path, dataset), "--base-model", str(base), "--max-length", "32",
            "--epochs", "1", "--batch-size", "8", "--cpu"]
    run_id = main([*args, "--stage", "all"])
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert "transformer.learning_rate" in run.data.params
    assert "transformer.test.f1_macro" in run.data.metrics
    assert "ensemble.test.f1_macro" in run.data.metrics
    bundle = tmp_path / "bundle"
    ensemble = SoftVotingEnsemble.from_manifest(bundle / "manifest.json")
    result = ensemble.predict("São Paulo não confirmou essa notícia!")
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["weights"] == {"lexical": 0.5, "transformer": 0.5}
    assert np.isfinite(ensemble.predict_proba(["pesquisa oficial", "boato secreto"])).all()
    from backend.config import Config
    from backend.services.analyzer import NewsAnalyzerService
    monkeypatch.setattr(Config, "MODEL_BACKEND", "ensemble")
    monkeypatch.setattr(Config, "ENSEMBLE_ENABLED", True)
    monkeypatch.setattr(Config, "ENSEMBLE_MANIFEST", bundle / "manifest.json")
    monkeypatch.setattr(Config, "TRANSFORMER_DEVICE", "cpu")
    analyzer = NewsAnalyzerService()
    assert analyzer.is_model_loaded() and analyzer.ensemble is not None
    analysis = analyzer.analyze({"text": "São Paulo não confirmou essa notícia!"})
    assert analysis["confidence_score"] == pytest.approx(result["ml_confidence_percent"])
    artifacts = mlflow.MlflowClient().list_artifacts(run_id, "bundle/transformer")
    assert "bundle/transformer/model.safetensors" in {a.path for a in artifacts}
    # Avaliação separada reaproveita exatamente os mesmos splits.
    separate_id = main([*args, "--stage", "ensemble"])
    assert "ensemble.validation.f1_macro" in mlflow.get_run(separate_id).data.metrics
    with pytest.raises(ValueError, match="Bundle ativo"):
        main([*args, "--stage", "lexical"])
    with pytest.raises(ValueError, match="mesmos dados"):
        main([*args, "--stage", "ensemble", "--seed", "7"])
    lexical_metadata_path = bundle / "lexical_metadata.json"
    lexical_metadata = json.loads(lexical_metadata_path.read_text())
    lexical_metadata["split_signature"] = "outro-split"
    lexical_metadata_path.write_text(json.dumps(lexical_metadata))
    with pytest.raises(ValueError, match="incompatíveis"):
        SoftVotingEnsemble.from_manifest(bundle / "manifest.json")
    metadata_path = bundle / "transformer" / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["labels"] = {"0": "real", "1": "fake"}
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mapeamento"):
        TransformerClassifier(bundle / "transformer")
