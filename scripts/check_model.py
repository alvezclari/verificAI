"""Verifica se o bundle de inferência foi baixado por inteiro pelo Git LFS."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_bundle(root=ROOT, verify_hashes=False):
    bundle = Path(root) / "models/mdeberta-ensemble"
    inventory = bundle / "checksums.json"
    if not inventory.is_file():
        raise ValueError("Inventário do modelo ausente; use a branch que contém o bundle publicado.")
    metadata = json.loads(inventory.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or not metadata.get("files"):
        raise ValueError("Inventário do modelo inválido.")
    for name, entry in metadata["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Caminho inválido no inventário: {name}")
        path = bundle / relative
        if not path.is_file():
            raise ValueError(f"Arquivo ausente: {name}. Execute git lfs pull.")
        with path.open("rb") as stream:
            if stream.read(len(LFS_POINTER)) == LFS_POINTER:
                raise ValueError(f"Ponteiro Git LFS em {name}; os pesos ainda não foram baixados. Execute git lfs pull.")
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Tamanho incorreto: {name}. Execute git lfs pull e tente novamente.")
        if verify_hashes and sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 não confere: {name}.")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != metadata["model_version"]:
        raise ValueError("Versão do modelo diverge do inventário.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Raiz do clone a verificar.")
    parser.add_argument("--verify-hashes", action="store_true", help="Confere SHA-256 de todos os arquivos, incluindo os pesos.")
    args = parser.parse_args()
    try:
        metadata = check_bundle(args.root, args.verify_hashes)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Modelo indisponível: {exc}\n")
    total = sum(entry["size"] for entry in metadata["files"].values())
    print(f"Bundle completo: {metadata['model_version']} ({len(metadata['files'])} arquivos, {total / 1024**3:.2f} GiB).")
    print("SHA-256 de todos os arquivos verificado." if args.verify_hashes else "Use --verify-hashes para conferir também os SHA-256.")


if __name__ == "__main__":
    main()
