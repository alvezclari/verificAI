import io
import json
from pathlib import Path
import sqlite3
import tarfile

import pytest

from scripts.transfer_artifacts import export_artifacts, import_artifacts


def source_tree(root):
    model = root / "models/mdeberta-ensemble"
    model.mkdir(parents=True)
    (model / "manifest.json").write_text(json.dumps({"version": "test-model"}))
    (model / "weights.bin").write_bytes(b"model-content")
    tracking = root / "mlruns/1/run/artifacts"
    tracking.mkdir(parents=True)
    (tracking / "weights.bin").write_bytes(b"model-content")
    with sqlite3.connect(root / "mlflow.db") as db:
        db.execute("CREATE TABLE experiments (artifact_location TEXT)")
        db.execute("CREATE TABLE runs (artifact_uri TEXT)")
        db.execute("INSERT INTO experiments VALUES (?)", ((root / "mlruns/1").as_uri(),))
        db.execute("INSERT INTO runs VALUES (?)", (str(tracking),))
    return root


def test_roundtrip_keeps_weights_and_relocates_tracking_without_shared_inodes(tmp_path):
    source = source_tree(tmp_path / "origin")
    archive = tmp_path / "transfer.tar.gz"
    manifest = export_artifacts(source, archive)
    with tarfile.open(archive) as package:
        assert package.getmember("mlruns/1/run/artifacts/weights.bin").islnk()
    target = tmp_path / "new-workspace"
    restored = import_artifacts(archive, target)
    assert restored["model_version"] == manifest["model_version"]
    weights = target / "models/mdeberta-ensemble/weights.bin"
    tracking_weights = target / "mlruns/1/run/artifacts/weights.bin"
    assert weights.read_bytes() == tracking_weights.read_bytes() == b"model-content"
    assert not weights.samefile(tracking_weights)
    with sqlite3.connect(target / "mlflow.db") as db:
        assert db.execute("SELECT artifact_location FROM experiments").fetchone()[0] == (target / "mlruns/1").as_uri()
        assert db.execute("SELECT artifact_uri FROM runs").fetchone()[0] == str(target / "mlruns/1/run/artifacts")
    assert (target / "handoff-import.json").is_file()
    assert weights.read_bytes() == (source / "models/mdeberta-ensemble/weights.bin").read_bytes()


def test_import_refuses_overwriting_existing_bundle(tmp_path):
    archive = tmp_path / "transfer.tar.gz"
    export_artifacts(source_tree(tmp_path / "origin"), archive)
    target = tmp_path / "target"
    (target / "models/mdeberta-ensemble").mkdir(parents=True)
    with pytest.raises(ValueError, match="já existem"):
        import_artifacts(archive, target)
    assert not (target / "mlflow.db").exists()


def test_checksum_failure_leaves_destination_untouched(tmp_path):
    archive = tmp_path / "transfer.tar.gz"
    export_artifacts(source_tree(tmp_path / "origin"), archive)
    Path(str(archive) + ".sha256").write_text("incorrect-digest transfer.tar.gz\n")
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="SHA-256"):
        import_artifacts(archive, target)
    assert not target.exists()


@pytest.mark.parametrize("name,digest", [("../outside", "unused"), ("data/file.csv", "incorrect")])
def test_member_integrity_and_paths_are_verified_before_publishing(tmp_path, name, digest):
    archive = tmp_path / "invalid.tar.gz"
    manifest = {"schema_version": 1, "source_root": str(tmp_path), "parts": ["data"],
                "files": {name: {"sha256": digest, "size": 3}}}
    with tarfile.open(archive, "w:gz") as package:
        for entry_name, data in [("handoff.json", json.dumps(manifest).encode()), (name, b"bad")]:
            member = tarfile.TarInfo(entry_name)
            member.size = len(data)
            package.addfile(member, io.BytesIO(data))
    target = tmp_path / "target"
    with pytest.raises(ValueError):
        import_artifacts(archive, target)
    assert not (target / "data").exists()
    assert not (tmp_path / "outside").exists()
