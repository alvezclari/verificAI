"""Transfere bundles, dados e MLflow local sem incluir ambientes virtuais."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PARTS = ("models/mdeberta-ensemble", "models/ensemble", "data", "mlruns", "mlflow.db")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def valid_name(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts
            and path.as_posix() == name
            and (name == "mlflow.db" or any(name.startswith(p + "/") for p in PARTS[:-1])))


def export_artifacts(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    model_manifest = root / "models/mdeberta-ensemble/manifest.json"
    if not model_manifest.is_file():
        raise ValueError("Bundle mDeBERTa ausente: restaure ou treine antes de exportar.")
    companions = [output, Path(str(output) + ".sha256"), Path(str(output) + ".manifest.json")]
    if any(p.exists() for p in companions):
        raise ValueError("Destino existente; escolha outro nome para a exportação.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verificai-export-") as work:
        files, parts = {}, []
        for part in PARTS:
            source = root / part
            if not source.exists():
                continue
            if source.is_symlink():
                raise ValueError(f"Link simbólico não permitido: {part}")
            if part == "mlflow.db":
                snapshot = Path(work) / "mlflow.db"
                origin = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
                destination = sqlite3.connect(snapshot)
                try:
                    origin.backup(destination)
                finally:
                    origin.close()
                    destination.close()
                files[part] = snapshot
            else:
                for path in sorted(source.rglob("*")):
                    if path.is_symlink():
                        raise ValueError(f"Link simbólico não permitido: {path}")
                    if path.is_file():
                        files[path.relative_to(root).as_posix()] = path
            if any(name == part or name.startswith(part + "/") for name in files):
                parts.append(part)
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(root),
            "git_commit": git_value(root, "rev-parse", "HEAD"),
            "git_branch": git_value(root, "branch", "--show-current"),
            "model_version": json.loads(model_manifest.read_text())["version"],
            "parts": parts,
            "files": {name: {"sha256": sha256(path), "size": path.stat().st_size}
                      for name, path in files.items()},
        }
        partial = Path(str(output) + ".partial")
        if partial.exists():
            raise ValueError("Exportação parcial existente; escolha outro destino.")
        try:
            with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
                content = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
                header = tarfile.TarInfo("handoff.json")
                header.size, header.mode = len(content), 0o644
                archive.addfile(header, io.BytesIO(content))
                stored = {}
                for name, path in files.items():
                    metadata = manifest["files"][name]
                    identity = (metadata["sha256"], metadata["size"])
                    header = tarfile.TarInfo(name)
                    header.mode = 0o644
                    if identity in stored:
                        # Evita armazenar os mesmos pesos novamente nas cópias do MLflow.
                        header.type, header.linkname = tarfile.LNKTYPE, stored[identity]
                        archive.addfile(header)
                    else:
                        header.size = metadata["size"]
                        with path.open("rb") as stream:
                            archive.addfile(header, stream)
                        stored[identity] = name
            partial.rename(output)
        finally:
            partial.unlink(missing_ok=True)
    companions[1].write_text(f"{sha256(output)}  {output.name}\n")
    companions[2].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def relocate_tracking(database, old_root, new_root):
    """Reescreve apenas localizações de artefatos; mantém IDs, parâmetros e métricas."""
    old_root, new_root = Path(old_root), Path(new_root)
    prefixes = [(str(old_root), str(new_root)), (old_root.as_uri(), new_root.as_uri())]
    connection = sqlite3.connect(database)
    try:
        with connection:
            for table, column in (("experiments", "artifact_location"), ("runs", "artifact_uri")):
                for rowid, value in connection.execute(f"SELECT rowid, {column} FROM {table}"):
                    for old, new in prefixes:
                        if value == old or value.startswith(old + "/"):
                            connection.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?",
                                               (new + value[len(old):], rowid))
                            break
    finally:
        connection.close()


def import_artifacts(archive_path, root):
    root, archive_path = Path(root).resolve(), Path(archive_path).resolve()
    archive_digest = sha256(archive_path)
    checksum = Path(str(archive_path) + ".sha256")
    if checksum.exists() and checksum.read_text().split()[:1] != [archive_digest]:
        raise ValueError("SHA-256 do pacote não confere.")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verificai-import-", dir=root) as work:
        staged = Path(work)
        with tarfile.open(archive_path, "r|gz") as archive:
            members = iter(archive)
            first = next(members, None)
            if first is None or first.name != "handoff.json" or not first.isfile() or first.size > 4_000_000:
                raise ValueError("Manifesto de transferência ausente ou inválido.")
            manifest = json.load(archive.extractfile(first))
            parts, expected = manifest.get("parts", []), manifest.get("files", {})
            if (manifest.get("schema_version") != 1 or not parts or not expected
                    or len(set(parts)) != len(parts) or any(p not in PARTS for p in parts)
                    or any(not valid_name(name) for name in expected)
                    or not Path(manifest.get("source_root", "")).is_absolute()
                    or any(not any(name == p or name.startswith(p + "/") for p in parts) for name in expected)
                    or any(not any(name == p or name.startswith(p + "/") for name in expected) for p in parts)):
                raise ValueError("Caminhos ou versão inválidos no manifesto de transferência.")
            if (root / "handoff-import.json").exists() or any((root / p).exists() for p in parts):
                raise ValueError("Artefatos já existem no destino; importe em um clone sem esses diretórios.")
            seen = set()
            for member in members:
                if member.name not in expected or member.name in seen:
                    raise ValueError("Arquivo inesperado ou duplicado no pacote.")
                target, entry = staged / member.name, expected[member.name]
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.isfile() and member.size == entry["size"]:
                    with archive.extractfile(member) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, 1024 * 1024)
                elif member.islnk() and member.linkname in seen and expected[member.linkname] == entry:
                    # Restaura cópias independentes: retreinar não altera artefatos históricos.
                    shutil.copyfile(staged / member.linkname, target)
                else:
                    raise ValueError("Tipo, tamanho ou referência de arquivo inválidos.")
                if target.stat().st_size != entry["size"] or sha256(target) != entry["sha256"]:
                    raise ValueError(f"SHA-256 não confere: {member.name}")
                seen.add(member.name)
            if seen != set(expected):
                raise ValueError("Pacote incompleto.")
        database = staged / "mlflow.db"
        if database.exists():
            relocate_tracking(database, manifest["source_root"], root)
        for part in parts:
            target = root / part
            target.parent.mkdir(parents=True, exist_ok=True)
            (staged / part).rename(target)
    audit = {**manifest, "imported_at": datetime.now(timezone.utc).isoformat(),
             "destination_root": str(root), "archive_sha256": archive_digest,
             "tracking_paths_relocated": "mlflow.db" in parts}
    (root / "handoff-import.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("--root", type=Path, default=ROOT, help="Raiz do repositório de origem/destino")
    parser.add_argument("--archive", type=Path, help="Obrigatório na importação; opcional na exportação")
    args = parser.parse_args()
    if args.action == "import" and args.archive is None:
        parser.error("import requer --archive")
    output = args.archive or args.root / "exports" / f"verificai-handoff-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.tar.gz"
    try:
        result = export_artifacts(args.root, output) if args.action == "export" else import_artifacts(output, args.root)
    except (ValueError, OSError, tarfile.TarError, sqlite3.Error, KeyError, TypeError) as error:
        parser.exit(1, f"Erro: {error}\n")
    print(json.dumps({"action": args.action, "archive": str(output),
                      "model_version": result["model_version"], "git_commit": result["git_commit"],
                      "files": len(result["files"]), "parts": result["parts"]}, indent=2))


if __name__ == "__main__":
    main()
