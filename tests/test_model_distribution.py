import hashlib
import json

import pytest

from scripts.check_model import check_bundle


def bundle_tree(root):
    bundle = root / "models/mdeberta-ensemble"
    bundle.mkdir(parents=True)
    files = {"manifest.json": b'{"version": "test-model"}', "weights.safetensors": b"test-weights"}
    for name, content in files.items():
        (bundle / name).write_bytes(content)
    (bundle / "checksums.json").write_text(json.dumps({
        "schema_version": 1, "model_version": "test-model",
        "files": {name: {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in files.items()},
    }), encoding="utf-8")
    return bundle


def test_downloaded_bundle_passes_integrity_check(tmp_path):
    bundle_tree(tmp_path)
    assert check_bundle(tmp_path, verify_hashes=True)["model_version"] == "test-model"


@pytest.mark.parametrize("content", [None, b"version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 12\n"])
def test_missing_weights_or_lfs_pointer_explains_how_to_download(tmp_path, content):
    bundle = bundle_tree(tmp_path)
    path = bundle / "weights.safetensors"
    if content is None:
        path.unlink()
    else:
        path.write_bytes(content)
    with pytest.raises(ValueError, match="git lfs pull"):
        check_bundle(tmp_path)


def test_changed_weights_with_same_size_fail_hash_check(tmp_path):
    bundle = bundle_tree(tmp_path)
    (bundle / "weights.safetensors").write_bytes(b"fake-weights")
    with pytest.raises(ValueError, match="SHA-256"):
        check_bundle(tmp_path, verify_hashes=True)
