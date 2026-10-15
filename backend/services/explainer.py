"""Explicação local por remoção de trechos, sobre a probabilidade final real."""

import math
import re
from time import perf_counter

BOUNDARIES = re.compile(r"[.!?…]+[\"'”’»)\]]*(?=\s|$)|\n+")
ABBREVIATIONS = {"dr", "dra", "sr", "sra", "prof", "profa", "etc", "ex", "art", "pág", "av", "aprox"}
MIN_EFFECT_PP = 0.5
LIMITATIONS = [
    "A explicação descreve mudanças na previsão; não verifica a veracidade dos fatos.",
    "Retirar um trecho também altera contexto e comprimento do texto. Os efeitos não são somáveis nem atribuições causais.",
]


def sentence_spans(text):
    """Segmentação conservadora; offsets referem-se ao texto original extraído."""
    start = 0
    for match in BOUNDARIES.finditer(text):
        if match.group().startswith("."):
            word = re.search(r"([\wÀ-ÿ]+)$", text[:match.start()])
            if word and (word[1].casefold() in ABBREVIATIONS or (len(word[1]) == 1 and word[1].isupper())):
                continue
        end = match.start() if match.group().startswith("\n") else match.end()
        left = start + len(text[start:end]) - len(text[start:end].lstrip())
        right = end - len(text[start:end]) + len(text[start:end].rstrip())
        if right > left:
            yield left, right
        start = match.end()
    left = start + len(text[start:]) - len(text[start:].lstrip())
    if text[left:].strip():
        yield left, len(text.rstrip())


def candidate_spans(title, body):
    candidates = []
    if len(title.split()) >= 3 and body.strip():
        candidates.append({"origem": "titulo", "inicio": 0, "fim": len(title), "texto": title})
    for start, end in sentence_spans(body):
        if len(body[start:end].split()) < 4:
            continue
        if not (title + body[:start] + body[end:]).strip():
            continue
        candidates.append({"origem": "corpo", "inicio": start, "fim": end, "texto": body[start:end]})
    return candidates


def select_candidates(candidates, budget):
    if len(candidates) <= budget:
        return candidates
    title = candidates[:1] if candidates[0]["origem"] == "titulo" else []
    body = candidates[len(title):]
    count = budget - len(title)
    if count == 0:
        return title
    indices = [round(i * (len(body) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
    return title + [body[i] for i in indices]


def percentage(probability):
    return f"{probability * 100:.1f}%".replace(".", ",")


def quote(text):
    text = " ".join(text.split())
    return text if len(text) <= 110 else text[:107].rstrip() + "…"


def make_summary(prediction, evaluations, status):
    labels = {"falsa": "possível desinformação", "verdadeira": "provavelmente verdadeira", "inconclusiva": "inconclusiva"}
    original = prediction["prob_fake"]
    sentences = [f"A estimativa da classe falsa foi {percentage(original)}, na faixa {labels.get(prediction.get('faixa'), 'informada pelo modelo')}."]
    if status == "insufficient_text":
        sentences.append("O texto não contém trechos suficientes para uma comparação por frases.")
    elif status == "no_effect":
        sentences.append("Os trechos comparados tiveram efeitos menores que 0,5 ponto percentual; esta análise não localizou uma influência expressiva em uma frase isolada.")
    else:
        for direction, verb in (("favorece_falsa", "caiu"), ("favorece_confiavel", "subiu")):
            items = [item for item in evaluations if item["direcao"] == direction]
            if items:
                strongest = max(items, key=lambda item: abs(item["efeito_pp"]))
                sentences.append(
                    f"Entre os trechos avaliados, ao retirar “{quote(strongest['texto'])}”, "
                    f"essa estimativa {verb} para {percentage(strongest['prob_fake_sem_trecho'])}."
                )
    return " ".join(sentences)


def explain_prediction(predictor, title, body, prediction, max_evaluations=8):
    started = perf_counter()
    # Mesmo uma configuração excessiva não permite um número ilimitado de inferências.
    budget = min(16, max(1, int(max_evaluations)))
    candidates = candidate_spans(title, body)
    selected = select_candidates(candidates, budget)
    documents = []
    for item in selected:
        if item["origem"] == "titulo":
            documents.append(("", body))
        else:
            documents.append((title, body[:item["inicio"]] + body[item["fim"]:]))
    evaluations = []
    if documents:
        if hasattr(predictor, "predict_documents_proba"):
            probabilities = predictor.predict_documents_proba(documents)
            if len(probabilities) != len(documents):
                raise ValueError("Quantidade de previsões XAI divergente.")
            scores = [float(row[0]) for row in probabilities]
        else:
            scores = [float(predictor.predict_document(t, b)["prob_fake"]) for t, b in documents]
        for item, score in zip(selected, scores):
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Probabilidade XAI inválida.")
            effect = (prediction["prob_fake"] - score) * 100
            direction = "favorece_falsa" if effect >= MIN_EFFECT_PP else ("favorece_confiavel" if effect <= -MIN_EFFECT_PP else "efeito_pequeno")
            evaluations.append({**item, "prob_fake_sem_trecho": score, "efeito_pp": effect, "direcao": direction})
    relevant = [item for item in evaluations if item["direcao"] != "efeito_pequeno"]
    status = "ok" if relevant else ("no_effect" if evaluations else "insufficient_text")
    highlighted = []
    for direction in ("favorece_falsa", "favorece_confiavel"):
        highlighted.extend(sorted(
            (item for item in relevant if item["direcao"] == direction),
            key=lambda item: abs(item["efeito_pp"]), reverse=True,
        )[:3])
    transformer = prediction.get("transformer") or {}
    limitations = list(LIMITATIONS)
    if len(candidates) > len(selected):
        limitations.append("A comparação usa uma amostra distribuída pelo texto; outros trechos também podem influenciar a previsão.")
    if transformer.get("truncated"):
        limitations.append(
            f"O ramo semântico usa até {transformer['max_length']} tokens. Retirar trechos pode trazer outras partes para essa janela; o ramo léxico considera o texto completo."
        )
    return {
        "metodo": "sentence_deletion", "scope": "final_prediction", "status": status,
        "resumo": make_summary(prediction, evaluations, status),
        "prob_fake_original": prediction["prob_fake"], "limiar_efeito_pp": MIN_EFFECT_PP,
        "trechos": highlighted, "avaliacoes": evaluations,
        "cobertura": {
            "trechos_identificados": len(candidates), "trechos_avaliados": len(selected),
            "parcial": len(selected) < len(candidates), "selecao": "titulo_e_amostragem_uniforme",
            "transformer_truncado": bool(transformer.get("truncated")),
            "transformer_tokens_usados": transformer.get("used_tokens"),
            "transformer_tokens_entrada": transformer.get("input_tokens"),
        },
        "limitacoes": limitations, "tempo_ms": round((perf_counter() - started) * 1000, 1),
    }
