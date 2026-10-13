"""Treino supervisionado, calibração reservada e importação do classificador atual."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from backend.ml.calibrated import CalibratedClassifier
from backend.ml.data import LABELS, fingerprint
from backend.ml.deep_ensemble import calibrated_blend
from backend.ml.ensemble import softmax, validate_weights
from backend.ml.training import log_evaluation, simple_params, train_transformer, write_json
from backend.ml.transformer import TransformerClassifier

ROOT = Path(__file__).resolve().parents[2]


def grouped_splits(labels, groups, seed=42):
    labels, groups = np.asarray(labels), np.asarray(groups)
    indices = np.arange(len(labels))
    train_cal_val, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(indices, labels, groups))
    remaining, cal_relative = next(StratifiedGroupKFold(8, shuffle=True, random_state=seed).split(
        train_cal_val, labels[train_cal_val], groups[train_cal_val]))
    calibration = train_cal_val[cal_relative]
    train_val = train_cal_val[remaining]
    train_relative, val_relative = next(StratifiedGroupKFold(7, shuffle=True, random_state=seed).split(
        train_val, labels[train_val], groups[train_val]))
    splits = {"train": train_val[train_relative], "validation": train_val[val_relative], "calibration": calibration, "test": test}
    for name, split in splits.items():
        if set(labels[split]) != {0, 1}:
            raise ValueError(f"Split {name} sem ambas as classes.")
    for name, split in splits.items():
        for other, other_split in splits.items():
            if name != other and set(groups[split]) & set(groups[other_split]):
                raise ValueError("Vazamento de grupos entre partições.")
    return {key: value.tolist() for key, value in splits.items()}


def fit_temperature(logits, labels):
    logits = np.asarray(logits)
    result = minimize_scalar(
        lambda log_t: log_loss(labels, softmax(logits / np.exp(log_t)), labels=[0, 1]),
        bounds=(np.log(0.2), np.log(5.0)), method="bounded",
    )
    if not result.success:
        raise ValueError("Calibração de temperatura falhou.")
    return float(np.exp(result.x))


def train(args):
    import mlflow
    import torch

    output = Path(args.output_dir).resolve()
    if (output / "manifest.json").exists() or (output / "transformer/training_metadata.json").exists():
        raise ValueError("Escolha outro diretório para uma nova versão de modelo.")
    with Path(args.data).open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    texts = [r["text"].strip() for r in records]
    labels = np.asarray([int(r["label"]) for r in records])
    groups = [r["group"] for r in records]
    if any(not text for text in texts):
        raise ValueError("Texto vazio no dataset.")
    if len(set(fingerprint(" ".join(t.casefold().split())) for t in texts)) != len(texts):
        raise ValueError("Execute a deduplicação antes de treinar.")
    splits = grouped_splits(labels, groups, args.seed)
    provenance = {
        "schema_version": 1, "labels": LABELS,
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "seed": args.seed, "splits": splits, "groups": groups,
        "split_policy": "StratifiedGroupKFold: aproximadamente 60/10/10/20; pares e URLs não cruzam partições.",
    }
    provenance["split_signature"] = fingerprint(provenance)
    source_file = Path(args.data).with_suffix(".provenance.json")
    if source_file.exists():
        provenance["source"] = json.loads(source_file.read_text())
    torch.set_num_threads(args.threads)
    lexical = CalibratedClassifier(args.classifier)
    weights = validate_weights(args.weights)
    output.mkdir(parents=True, exist_ok=True)
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    def samples(name):
        ix = splits[name]
        return [texts[i] for i in ix], labels[ix]
    with mlflow.start_run(run_name="mdeberta-supervised-calibrated-ensemble") as run:
        mlflow.log_params(simple_params(vars(args)))
        mlflow.log_params({f"data.{key}_count": len(value) for key, value in splits.items()})
        mlflow.log_param("data.sha256", provenance["dataset_sha256"])
        mlflow.log_param("lexical.sha256", lexical.sha256)
        mlflow.set_tag("lexical.imported", "true")
        mlflow.log_dict(lexical.manifest, "imported_classifier_manifest.json")
        mlflow.log_dict(provenance, "data_provenance.json")
        mlflow.log_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())), "environment.txt")
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
        if git.returncode == 0:
            mlflow.set_tag("git_commit", git.stdout.strip())
        semantic_path = output / "transformer"
        train_transformer(args, *samples("train"), *samples("validation"), semantic_path, mlflow)
        metadata = {
            "schema_version": 1, "labels": LABELS, "task": "supervised_fake_news",
            "max_length": args.max_length, "temperature": 1.0, "run_id": run.info.run_id,
            "base_model": args.base_model, "source_model": args.source_model,
            "source_revision": args.source_revision, "split_signature": provenance["split_signature"],
            "dataset_sha256": provenance["dataset_sha256"],
            "freeze_encoder_layers": args.freeze_encoder_layers,
        }
        write_json(semantic_path / "training_metadata.json", metadata)
        device = "cpu" if args.cpu else args.device
        semantic = TransformerClassifier(semantic_path, device=device, batch_size=args.batch_size)
        cal_texts, cal_labels = samples("calibration")
        cal_logits = semantic.predict_logits(cal_texts)
        temperature = fit_temperature(cal_logits, cal_labels)
        semantic.temperature = temperature
        metadata["temperature"] = temperature
        write_json(semantic_path / "training_metadata.json", metadata)
        lexical_cal = lexical.predict_proba(cal_texts)
        semantic_cal = softmax(cal_logits / temperature)
        raw_cal = lexical_cal * weights[0] + semantic_cal * weights[1]
        ensemble_temperature = fit_temperature(np.log(np.clip(raw_cal, 1e-12, 1)), cal_labels)
        mlflow.log_params({"calibration.transformer_temperature": temperature, "calibration.ensemble_temperature": ensemble_temperature})
        evaluation = {"protocol": "experimental_grouped_holdout", "production_ready": False, "splits": {},
                      "note": "Métricas do experimento; não medem o acerto de uma URL individual."}
        for split in ("validation", "calibration", "test"):
            x, y = samples(split)
            lexical_probs = lexical_cal if split == "calibration" else lexical.predict_proba(x)
            semantic_probs = semantic_cal if split == "calibration" else semantic.predict_proba(x)
            ensemble_probs = calibrated_blend(lexical_probs, semantic_probs, weights, ensemble_temperature)
            evaluation["splits"][split] = {
                "lexical": log_evaluation(mlflow, "imported_lexical", split, y, lexical_probs),
                "transformer": log_evaluation(mlflow, "mdeberta", split, y, semantic_probs),
                "ensemble": log_evaluation(mlflow, "deep_ensemble", split, y, ensemble_probs),
            }
            # O banco contém apenas falsos: sua métrica específica é recall da classe falsa.
            origin_indices = [j for j, i in enumerate(splits[split]) if records[i]["origin"] == "banco_fake"]
            if origin_indices:
                for component, probs in (("lexical", lexical_probs), ("transformer", semantic_probs), ("ensemble", ensemble_probs)):
                    mlflow.log_metric(f"banco_fake.{split}.{component}.fake_recall", float((probs[origin_indices, 0] >= 0.5).mean()))
        shutil.copy2(args.classifier, output / "classificador.joblib")
        write_json(output / "data_provenance.json", provenance)
        manifest = {
            "schema_version": 2, "labels": LABELS,
            "version": "mdeberta-ensemble-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
            "calibrated_path": "classificador.joblib", "calibrated_sha256": lexical.sha256,
            "transformer_path": "transformer", "weights": weights.tolist(),
            "temperature": ensemble_temperature, "thresholds": {"verdadeira_max": 0.35, "inconclusiva_max": 0.65},
            "split_signature": provenance["split_signature"], "run_id": run.info.run_id,
            "evaluation": evaluation,
            "limitations": provenance.get("source", {}).get("limitations", []) + [
                "Modelo experimental, com truncamento de notícias e possíveis diferenças por fonte/época.",
                "O classificador importado tem partições históricas próprias; independência cruzada completa não foi certificada.",
                "Contribuições exibidas são do ramo léxico, não atribuições causais do ensemble.",
            ],
        }
        mlflow.log_dict(manifest, "bundle/manifest.json")
        mlflow.log_artifact(str(output / "classificador.joblib"), "bundle")
        mlflow.log_artifact(str(output / "data_provenance.json"), "bundle")
        mlflow.log_artifacts(str(semantic_path), "bundle/transformer")
        write_json(output / "manifest.json.tmp", manifest)
        (output / "manifest.json.tmp").replace(output / "manifest.json")
        print(json.dumps({"run_id": run.info.run_id, "output": str(output), "test": evaluation["splits"]["test"]}, indent=2), flush=True)
        return run.info.run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/mdeberta_news.csv")
    parser.add_argument("--classifier", default="models/classificador.joblib")
    parser.add_argument("--output-dir", default="models/mdeberta-ensemble")
    parser.add_argument("--base-model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--source-model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--replace-classification-head", action="store_true")
    parser.add_argument("--freeze-encoder-layers", type=int, default=10)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default="verificai-mdeberta-imported-ensemble")
    parser.add_argument("--calibrate-existing", action="store_true", help="Seleciona pesos na validação e ajusta intercepto na calibração de um bundle concluído")
    args = parser.parse_args(argv)
    if args.calibrate_existing:
        from backend.ml.deep_calibration import calibrate_existing
        return calibrate_existing(args)
    return train(args)
