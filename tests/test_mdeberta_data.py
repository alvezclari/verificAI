import csv
import json
import zipfile

import pytest

from scripts.prepare_mdeberta_data import prepare


def test_corpus_keeps_pairs_raw_text_and_explicit_user_label_source(tmp_path):
    archive = tmp_path / "fakebr.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("Fake.br-Corpus-master/", "")
        for label in ("fake", "true"):
            output.writestr(f"Fake.br-Corpus-master/full_texts/{label}/1.txt", f"Texto original {label}: não há confirmação, 50%!")
            output.writestr(f"Fake.br-Corpus-master/full_texts/{label}-meta-information/1-meta.txt", f"autor\nhttps://example.com/{label}\npolitica\n2020-01-01")
    bank = tmp_path / "banco.csv"
    with bank.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Título", "Corpo da notícia", "Link", "Fonte / Portal"])
        writer.writeheader()
        writer.writerow({"Título": "Título de exemplo", "Corpo da notícia": "Corpo com negação: não aconteceu.", "Link": "https://example.com/banco", "Fonte / Portal": "fonte"})
    target = tmp_path / "dataset.csv"
    report = prepare(archive, bank, target)
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["group"] == rows[1]["group"] == "fakebr-pair-1"
    assert {r["label"] for r in rows} == {"0", "1"}
    assert rows[-1]["label_source"] == "user_confirmed_false"
    assert "não" in rows[-1]["text"] and "50%" in rows[0]["text"]
    assert report["classes"] == {"fake": 2, "real": 1}
    assert json.loads(target.with_suffix(".provenance.json").read_text())["output_sha256"] == report["output_sha256"]
