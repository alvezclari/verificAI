"""Combina textos originais Fake.Br e banco_fake confirmado pelo responsável."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(fakebr_zip, banco_fake, output, revision_file=None):
    records = []
    with zipfile.ZipFile(fakebr_zip) as archive:
        root = archive.namelist()[0].split("/")[0]
        for label, folder in ((0, "fake"), (1, "true")):
            paths = sorted(n for n in archive.namelist() if re.search(rf"/full_texts/{folder}/\d+\.txt$", n))
            for path in paths:
                identifier = Path(path).stem
                text = archive.read(path).decode("utf-8-sig").strip()
                metadata_path = f"{root}/full_texts/{folder}-meta-information/{identifier}-meta.txt"
                metadata = archive.read(metadata_path).decode("utf-8-sig").splitlines()
                url = metadata[1].strip()
                records.append({
                    "text": text, "label": label, "title": "", "body": text, "url": url,
                    "group": f"fakebr-pair-{identifier}", "origin": "fakebr",
                    "label_source": "original_corpus", "source": url.split("/")[2] if "://" in url else "unknown",
                })
    with Path(banco_fake).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            title, body, url = row["Título"].strip(), row["Corpo da notícia"].strip(), row["Link"].strip()
            if not body or not url:
                raise ValueError("banco_fake contém notícia sem corpo ou URL.")
            records.append({
                "text": f"{title}\n\n{body}".strip(), "label": 0, "title": title, "body": body,
                "url": url, "group": "banco-" + hashlib.sha256(url.encode()).hexdigest(),
                "origin": "banco_fake", "label_source": "user_confirmed_false", "source": row["Fonte / Portal"],
            })
    # O mesmo texto e a mesma URL nunca devem aparecer em partições diferentes.
    seen_text, seen_url, unique = {}, {}, []
    for row in records:
        key = hashlib.sha256(" ".join(row["text"].casefold().split()).encode()).hexdigest()
        previous = seen_text.get(key) or seen_url.get(row["url"])
        if previous:
            if previous["label"] != row["label"]:
                raise ValueError("Conteúdo/URL com rótulos conflitantes; revisão necessária.")
            continue
        seen_text[key] = row
        seen_url[row["url"]] = row
        unique.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unique[0]))
        writer.writeheader()
        writer.writerows(unique)
    provenance = {
        "schema_version": 1, "fakebr_source": "https://github.com/roneysco/Fake.br-Corpus",
        "fakebr_archive_sha256": digest(fakebr_zip), "banco_fake_sha256": digest(banco_fake),
        "output_sha256": digest(output), "records": len(unique), "duplicates_removed": len(records) - len(unique),
        "classes": {"fake": sum(r["label"] == 0 for r in unique), "real": sum(r["label"] == 1 for r in unique)},
        "banco_fake_label_origin": "Responsável confirmou que todos os registros são falsos em 13/10/2026.",
        "split_policy": "Grupos de pares Fake.Br e URLs; deduplicação textual exata antes dos splits.",
        "limitations": [
            "Rótulos do banco_fake fornecidos pelo responsável; não houve nova checagem factual automatizada.",
            "As fontes/épocas podem correlacionar-se com os rótulos; avaliação experimental.",
            "Agrupamento de pares e duplicatas exatas não exclui todas as paráfrases.",
        ],
    }
    if revision_file:
        provenance["fakebr_revision"] = json.loads(Path(revision_file).read_text())["sha"]
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fakebr-zip", default="data/raw/fakebr.zip")
    parser.add_argument("--banco-fake", default="banco_fake.csv")
    parser.add_argument("--output", default="data/processed/mdeberta_news.csv")
    parser.add_argument("--revision-file", default="data/raw/fakebr_revision.json")
    parser.add_argument("--confirm-banco-false", action="store_true", required=True)
    args = parser.parse_args()
    prepare(args.fakebr_zip, args.banco_fake, args.output, args.revision_file)
