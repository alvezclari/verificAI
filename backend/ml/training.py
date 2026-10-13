"""Treino separado ou integrado, com avaliação e artefatos no MLflow."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from backend.ml.data import LABELS, load_records, split_records
from backend.ml.ensemble import LexicalClassifier, SoftVotingEnsemble, softmax, validate_weights

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_lexical_model(seed=42, max_features=5000, c=1.0, max_iter=1000):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=max_features)),
        ("clf", LogisticRegression(C=c, max_iter=max_iter, random_state=seed)),
    ])


def metrics_for(labels, probabilities):
    predicted = (probabilities[:, 1] >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "f1_macro": f1_score(labels, predicted, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, predicted, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predicted, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities[:, 1]),
        "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(labels, probabilities[:, 1]),
    }


def log_evaluation(mlflow, component, split, labels, probabilities):
    metrics = metrics_for(labels, probabilities)
    mlflow.log_metrics({f"{component}.{split}.{k}": float(v) for k, v in metrics.items()})
    predicted = (probabilities[:, 1] >= 0.5).astype(int)
    mlflow.log_dict({
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels, predicted, labels=[0, 1], target_names=["fake", "confiavel"],
            output_dict=True, zero_division=0,
        ),
    }, f"evaluation/{component}_{split}.json")
    return metrics


def simple_params(params):
    return {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in params.items()}


def train_transformer(args, train_texts, train_labels, val_texts, val_labels, output, mlflow):
    import torch
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainerCallback, TrainingArguments, set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, revision=args.revision, num_labels=2,
        ignore_mismatched_sizes=getattr(args, "replace_classification_head", False),
        id2label={0: "fake", 1: "confiavel"}, label2id={"fake": 0, "confiavel": 1},
    )
    frozen_layers = getattr(args, "freeze_encoder_layers", 0)
    if frozen_layers:
        encoder_layers = model.base_model.encoder.layer
        if not 0 <= frozen_layers <= len(encoder_layers):
            raise ValueError("Quantidade de camadas congeladas inválida.")
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
        for layer in encoder_layers[frozen_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    mlflow.log_params({
        "transformer.trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "transformer.total_parameters": sum(p.numel() for p in model.parameters()),
    })
    if args.max_length > getattr(model.config, "max_position_embeddings", args.max_length):
        raise ValueError("max_length excede a capacidade do modelo.")
    mlflow.log_param("transformer.resolved_revision", getattr(model.config, "_commit_hash", None) or "local")
    mlflow.log_dict(model.config.to_dict(), "transformer_config.json")

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, max_length=args.max_length)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            encoding = {key: values[index] for key, values in self.encodings.items()}
            encoding["labels"] = int(self.labels[index])
            return encoding

    class TrackingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            mlflow.log_metrics({f"trainer.{k}": float(v) for k, v in (logs or {}).items()
                                if isinstance(v, (int, float)) and np.isfinite(v)}, step=state.global_step)

    # Checkpoints intermediários não entram no bundle de inferência.
    with tempfile.TemporaryDirectory(prefix="verificai-trainer-") as work_dir:
        training_args = TrainingArguments(
            output_dir=work_dir, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio, gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed, data_seed=args.seed, eval_strategy="epoch", save_strategy="epoch",
            logging_strategy="steps", logging_steps=10, load_best_model_at_end=True,
            metric_for_best_model="f1_macro", greater_is_better=True, save_total_limit=1,
            report_to=[], use_cpu=args.cpu,
        )
        mlflow.log_params({f"transformer.{k}": v for k, v in simple_params(training_args.to_dict()).items()})
        optimizer = None
        head_learning_rate = getattr(args, "head_learning_rate", None)
        if head_learning_rate is not None:
            backbone_ids = {id(p) for p in model.base_model.parameters()}
            optimizer = torch.optim.AdamW([
                {"params": [p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids], "lr": args.learning_rate},
                {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids], "lr": head_learning_rate},
            ], weight_decay=args.weight_decay)
        trainer = Trainer(
            model=model, args=training_args, processing_class=tokenizer,
            train_dataset=TextDataset(train_texts, train_labels),
            eval_dataset=TextDataset(val_texts, val_labels),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=lambda result: metrics_for(result.label_ids, softmax(result.predictions)),
            callbacks=[TrackingCallback()],
            optimizers=(optimizer, None),
        )
        trainer.train()
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        mlflow.log_dict(trainer.state.log_history, "trainer_history.json")


def run_training(args, texts, labels, source):
    # Import obrigatório: falhas de rastreamento interrompem o treino.
    import mlflow

    weights = validate_weights(args.weights)
    splits, provenance = split_records(texts, labels, args.seed, args.validation_size, args.test_size)
    provenance["source"] = source
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if args.stage != "ensemble" and manifest.exists():
        raise ValueError("Bundle ativo já existe; escolha outro --output-dir para uma nova versão.")
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    groups = {name: ([texts[i] for i in ix], np.asarray([labels[i] for i in ix])) for name, ix in splits.items()}
    with mlflow.start_run(run_name=f"{args.stage}-seed-{args.seed}") as run:
        params = vars(args).copy()
        params["weights"] = weights.tolist()
        mlflow.log_params(simple_params(params))
        mlflow.log_params({f"data.{name}_count": len(ix) for name, ix in splits.items()})
        mlflow.log_param("data.sha256", provenance["dataset_sha256"])
        mlflow.log_param("data.split_signature", provenance["split_signature"])
        mlflow.log_dict(provenance, "data_provenance.json")
        mlflow.set_tag("label_mapping", json.dumps(LABELS))
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
        if git.returncode == 0:
            mlflow.set_tag("git_commit", git.stdout.strip())
        mlflow.log_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())), "environment.txt")
        metadata = {
            "schema_version": 1, "labels": LABELS, "split_signature": provenance["split_signature"],
            "dataset_sha256": provenance["dataset_sha256"], "run_id": run.info.run_id,
        }
        if args.stage in ("lexical", "all"):
            lexical = build_lexical_model(args.seed, args.max_features, args.c, args.max_iter)
            mlflow.log_params({f"lexical.{k}": v for k, v in simple_params(lexical.get_params()).items()})
            lexical.fit(*groups["train"])
            joblib.dump(lexical, output / "lexical.joblib")
            write_json(output / "lexical_metadata.json", metadata)
            mlflow.log_artifact(str(output / "lexical.joblib"), "bundle")
            mlflow.log_artifact(str(output / "lexical_metadata.json"), "bundle")
            for split in ("validation", "test"):
                x, y = groups[split]
                log_evaluation(mlflow, "lexical", split, y, lexical.predict_proba(x))
        if args.stage in ("transformer", "all"):
            semantic_path = output / "transformer"
            # O marcador de treino só é escrito após o fine-tuning terminar.
            if (semantic_path / "training_metadata.json").exists():
                raise ValueError("Checkpoint já existe; escolha outro --output-dir.")
            train_transformer(args, *groups["train"], *groups["validation"], semantic_path, mlflow)
            write_json(semantic_path / "training_metadata.json", {
                **metadata, "max_length": args.max_length, "base_model": args.base_model,
                "revision": args.revision,
            })
            mlflow.log_artifacts(str(semantic_path), "bundle/transformer")
            from backend.ml.transformer import TransformerClassifier
            semantic = TransformerClassifier(semantic_path)
            for split in ("validation", "test"):
                x, y = groups[split]
                log_evaluation(mlflow, "transformer", split, y, semantic.predict_proba(x))
        if args.stage in ("all", "ensemble"):
            from backend.ml.transformer import TransformerClassifier
            component_runs = {}
            for component, path in {
                "lexical": output / "lexical_metadata.json",
                "transformer": output / "transformer" / "training_metadata.json",
            }.items():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("split_signature") != provenance["split_signature"] or saved.get("labels") != LABELS:
                    raise ValueError("Componentes devem usar os mesmos dados, rótulos e splits.")
                component_runs[component] = saved["run_id"]
            semantic = semantic if args.stage == "all" else TransformerClassifier(output / "transformer")
            ensemble = SoftVotingEnsemble(
                LexicalClassifier(joblib.load(output / "lexical.joblib")),
                semantic, weights,
            )
            for split in ("validation", "test"):
                x, y = groups[split]
                lexical_probs, semantic_probs = ensemble.component_probabilities(x)
                if args.stage == "ensemble":
                    log_evaluation(mlflow, "lexical", split, y, lexical_probs)
                    log_evaluation(mlflow, "transformer", split, y, semantic_probs)
                log_evaluation(mlflow, "ensemble", split, y, lexical_probs * weights[0] + semantic_probs * weights[1])
            write_json(output / "data_provenance.json", provenance)
            # Publica o manifesto só quando ambos os componentes e avaliações estão prontos.
            bundle_manifest = {
                **metadata, "lexical_path": "lexical.joblib", "transformer_path": "transformer",
                "weights": weights.tolist(), "component_run_ids": component_runs,
            }
            mlflow.log_dict(bundle_manifest, "bundle/manifest.json")
            mlflow.log_artifact(str(output / "data_provenance.json"), "bundle")
            if args.stage == "ensemble":
                mlflow.log_artifact(str(output / "lexical.joblib"), "bundle")
                mlflow.log_artifact(str(output / "lexical_metadata.json"), "bundle")
                mlflow.log_artifacts(str(output / "transformer"), "bundle/transformer")
            temporary_manifest = output / "manifest.json.tmp"
            write_json(temporary_manifest, bundle_manifest)
            temporary_manifest.replace(manifest)
        print(f"MLflow run: {run.info.run_id}; artefatos: {output}")
        return run.info.run_id


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["lexical", "transformer", "all", "ensemble"], default="all")
    parser.add_argument("--data", required=True, help="CSV UTF-8 ou ZIP com --zip-member")
    parser.add_argument("--zip-member")
    parser.add_argument("--text-column", default="texto_bruto")
    parser.add_argument("--label-column", default="Classe")
    parser.add_argument("--real-label", required=True, help="Valor original correspondente à notícia real")
    parser.add_argument("--output-dir", default=str(ROOT / "models" / "ensemble"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--base-model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--revision", default="main", help="Use hash de commit do Hub para reprodução estrita")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--weights", nargs=2, type=float, default=[0.5, 0.5], metavar=("LEXICAL", "TRANSFORMER"))
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default="verificai-ensemble")
    args = parser.parse_args(argv)
    if (args.max_length <= 0 or args.epochs <= 0 or args.batch_size <= 0
            or args.gradient_accumulation_steps <= 0 or args.learning_rate <= 0):
        parser.error("Hiperparâmetros de comprimento, épocas, batch e aprendizado devem ser positivos.")
    return args


def main(argv=None):
    args = parse_args(argv)
    texts, labels, source = load_records(args.data, args.text_column, args.label_column, args.real_label, args.zip_member)
    return run_training(args, texts, labels, source)
