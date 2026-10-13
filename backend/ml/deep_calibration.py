"""Seleção na validação e correção de viés na reserva de calibração."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import f1_score, log_loss

from backend.ml.deep_ensemble import CalibratedDeepEnsemble, calibrated_blend
from backend.ml.training import log_evaluation, simple_params, write_json


def fit_bias_calibration(probabilities, reliable_labels):
    probabilities = np.clip(np.asarray(probabilities), 1e-12, 1 - 1e-12)
    log_odds_fake = np.log(probabilities[:, 0] / probabilities[:, 1])
    fake_labels = 1 - np.asarray(reliable_labels)
    def objective(parameters):
        log_temperature, bias = parameters
        scaled = log_odds_fake / np.exp(log_temperature)
        logits = scaled + bias
        error = expit(logits) - fake_labels
        loss = np.mean(np.logaddexp(0, logits) - fake_labels * logits)
        return loss, np.array([np.mean(error * -scaled), np.mean(error)])
    result = minimize(objective, [0.0, 0.0], jac=True, method="L-BFGS-B",
                      bounds=[(np.log(0.1), np.log(10)), (-10, 10)])
    if not result.success:
        raise ValueError("Calibração com intercepto não convergiu.")
    return float(np.exp(result.x[0])), float(result.x[1])


def select_combination(lexical_cal, semantic_cal, labels_cal, lexical_val, semantic_val, labels_val):
    candidates = []
    for lexical_weight in (0.1, 0.25, 0.5, 0.75, 0.9):
        weights = [lexical_weight, 1 - lexical_weight]
        raw_cal = lexical_cal * weights[0] + semantic_cal * weights[1]
        temperature, bias = fit_bias_calibration(raw_cal, labels_cal)
        prediction = calibrated_blend(lexical_val, semantic_val, weights, temperature, bias)
        candidates.append({
            "weights": weights, "temperature": temperature, "bias": bias,
            "validation_f1_macro": float(f1_score(labels_val, prediction[:, 1] >= 0.5, average="macro")),
            "validation_log_loss": float(log_loss(labels_val, prediction, labels=[0, 1])),
        })
    selected = max(candidates, key=lambda item: (item["validation_f1_macro"], -item["validation_log_loss"]))
    return selected, candidates


def calibrate_existing(args):
    import mlflow
    from backend.ml.deep_training import ROOT

    root = Path(args.output_dir).resolve()
    manifest_path = root / "manifest.json"
    previous = json.loads(manifest_path.read_text())
    provenance = json.loads((root / "data_provenance.json").read_text())
    if hashlib.sha256(Path(args.data).read_bytes()).hexdigest() != provenance["dataset_sha256"]:
        raise ValueError("Dataset diverge do treinamento.")
    with Path(args.data).open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    splits = provenance["splits"]
    ensemble = CalibratedDeepEnsemble.from_manifest(manifest_path, device="cpu" if args.cpu else args.device)
    mlflow.set_tracking_uri(args.tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="mdeberta-validation-weight-selection-and-bias-calibration") as run:
        mlflow.set_tag("training_run_id", previous["run_id"])
        mlflow.log_params({"weight_selection_split": "validation", "calibration_fit_split": "calibration",
                           "criterion": "f1_macro_then_log_loss", "dataset_sha256": provenance["dataset_sha256"]})
        mlflow.log_params({f"lexical.{key}": value for key, value in simple_params(ensemble.lexical.motor.artefato.pipeline.get_params()).items()})
        mlflow.log_dict(previous, "previous_manifest.json")
        probabilities = {}
        def samples(name):
            indices = splits[name]
            return [records[i]["text"] for i in indices], np.array([int(records[i]["label"]) for i in indices])
        for split in ("calibration", "validation"):
            x, y = samples(split)
            lexical, semantic = ensemble.component_probabilities(x)
            probabilities[split] = (lexical, semantic, y)
        chosen, candidates = select_combination(*probabilities["calibration"], *probabilities["validation"])
        mlflow.log_dict({"selected": chosen, "candidates": candidates}, "weight_selection.json")
        mlflow.log_params({"weights": str(chosen["weights"]), "temperature": chosen["temperature"], "bias": chosen["bias"]})
        evaluation = {"protocol": "experimental_grouped_holdout_validation_selected_weights", "production_ready": False,
                      "note": "Métricas do experimento; não medem o acerto de uma URL individual.", "splits": {}}
        # O teste só entra depois da seleção da combinação na validação.
        for split in ("validation", "calibration", "test"):
            if split not in probabilities:
                x, y = samples(split)
                lexical, semantic = ensemble.component_probabilities(x)
                probabilities[split] = (lexical, semantic, y)
            lexical, semantic, y = probabilities[split]
            combined = calibrated_blend(lexical, semantic, chosen["weights"], chosen["temperature"], chosen["bias"])
            evaluation["splits"][split] = {
                "lexical": log_evaluation(mlflow, "imported_lexical", split, y, lexical),
                "transformer": log_evaluation(mlflow, "mdeberta", split, y, semantic),
                "ensemble": log_evaluation(mlflow, "deep_ensemble", split, y, combined),
            }
            selected = [j for j, i in enumerate(splits[split]) if records[i]["origin"] == "banco_fake"]
            if selected:
                bank = {}
                for name, values in (("lexical", lexical), ("transformer", semantic), ("ensemble", combined)):
                    bank[name] = {
                        "fake_recall_at_0_5": float((values[selected, 0] >= 0.5).mean()),
                        "fake_band_rate": float((values[selected, 0] > 0.65).mean()),
                        "inconclusive_rate": float(((values[selected, 0] > 0.35) & (values[selected, 0] <= 0.65)).mean()),
                    }
                evaluation.setdefault("banco_fake", {})[split] = {"count": len(selected), "components": bank}
                mlflow.log_dict(bank, f"evaluation/banco_fake_{split}.json")
        result = {
            **previous, "version": "mdeberta-ensemble-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
            "training_run_id": previous.get("training_run_id", previous["run_id"]), "run_id": run.info.run_id,
            "weights": chosen["weights"], "temperature": chosen["temperature"], "bias": chosen["bias"],
            "calibration_method": "sigmoid_on_reserved_calibration_split", "weight_selection": {"split": "validation", "candidates": candidates},
            "evaluation": evaluation,
        }
        mlflow.log_dict(result, "bundle/manifest.json")
        mlflow.log_artifact(str(root / "classificador.joblib"), "bundle")
        mlflow.log_artifact(str(root / "data_provenance.json"), "bundle")
        mlflow.log_artifacts(str(root / "transformer"), "bundle/transformer")
        write_json(root / "manifest.json.tmp", result)
        (root / "manifest.json.tmp").replace(manifest_path)
        print(json.dumps({"run_id": run.info.run_id, "selected": chosen, "test": evaluation["splits"]["test"]}, indent=2), flush=True)
        return run.info.run_id
