"""Verificação conservadora por trechos reais; nunca modifica probabilidades textuais."""

import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from ml.dataset import fonte_url
from ml.preprocessing import preparar_texto
from scraper.extractor import Documento, extrair_url
from scraper.providers import SemBusca

VERSAO_VERIFICACAO = "1.0-conservadora"
_OPERACOES = ThreadPoolExecutor(max_workers=8, thread_name_prefix="evidencias")


def _no_prazo(funcao, limite_global, *args, **kwargs):
    restante = limite_global - time.monotonic()
    if restante <= 0:
        raise TimeoutError("Prazo global atingido.")
    tarefa = _OPERACOES.submit(funcao, *args, **kwargs)
    try:
        return tarefa.result(timeout=restante)
    except TimeoutError:
        tarefa.cancel()
        raise


def dividir_alegacoes(texto: str, limite=8) -> tuple[list[dict], bool]:
    sentencas = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", texto) if s.strip()]
    resultados = []
    for sentenca in sentencas[:limite]:
        normal = preparar_texto(None, sentenca)
        motivo = None
        if "?" in sentenca:
            motivo = "pergunta"
        elif re.search(r'["“”«»]|\b(disse|segundo|afirmou|declarou|alegou)\b', normal):
            motivo = "citacao_ou_atribuicao"
        elif re.search(
            r"\b(será|serão|haverá|deverá|deverão|amanhã|previsão|prevê|vai acontecer)\b", normal
        ):
            motivo = "previsao_futura"
        elif re.match(r"^(veja|confira|saiba|leia|assista|entenda)\b", normal):
            motivo = "chamada_editorial"
        elif (
            sentenca.endswith(("...", "…", ":")) or sentenca[-1] not in ".!?" or len(sentenca) > 800
        ):
            motivo = "trecho_truncado"
        elif len(normal.split()) < 6:
            motivo = "contexto_insuficiente"
        resultados.append(
            {
                "alegacao": sentenca[:800],
                "verificavel": motivo is None,
                "motivo_nao_verificavel": motivo,
                "status": "evidencia_insuficiente",
                "evidencias": [],
            }
        )
    return resultados, len(sentencas) > limite


def comparar_alegacoes(alegacao: str, trecho: str) -> tuple[str, list[str]]:
    """Só confirma coincidência literal ou oposição explícita de uma negação simples."""
    a, b = preparar_texto(None, alegacao), preparar_texto(None, trecho)
    a, b = a.rstrip(".! "), b.rstrip(".! ")
    if a == b:
        return "sustentada", ["Coincidência textual não comprova a verdade factual do conteúdo."]
    numeros_a, numeros_b = re.findall(r"\d+(?:[.,/]\d+)*", a), re.findall(r"\d+(?:[.,/]\d+)*", b)
    if numeros_a != numeros_b:
        return "evidencia_insuficiente", [
            "Números ou datas divergem; contexto temporal não confirmado."
        ]
    negacoes = r"\b(não|nunca|jamais)\b"
    na, nb = re.findall(negacoes, a), re.findall(negacoes, b)
    sem_a = " ".join(re.sub(negacoes, "", a).split())
    sem_b = " ".join(re.sub(negacoes, "", b).split())
    if sem_a == sem_b and bool(na) != bool(nb) and len(na) + len(nb) == 1:
        if not re.search(r"não (só|apenas|necessariamente)", a + " " + b):
            return "refutada", ["Oposição textual explícita; não substitui revisão de contexto."]
    return "evidencia_insuficiente", ["Não há correspondência textual inequívoca."]


def _copias(a: str, b: str) -> bool:
    a, b = preparar_texto(None, a), preparar_texto(None, b)
    if a == b:
        return True

    def shingles(texto):
        palavras = texto[:100000].split()
        return {tuple(palavras[i : i + 5]) for i in range(len(palavras) - 4)}

    sa, sb = shingles(a), shingles(b)
    return bool(sa and sb and len(sa & sb) / min(len(sa), len(sb)) >= 0.8)


def verificar(
    documento: Documento,
    evidence_urls=(),
    reference_date=None,
    search=True,
    provedores=None,
    extrator=extrair_url,
    prazo_segundos=30,
) -> dict:
    inicio = time.monotonic()
    prazo = inicio + min(30, prazo_segundos)
    referencia = reference_date or date.today()
    resultados, parcial = dividir_alegacoes(documento.corpo)
    provedores = provedores or [SemBusca()]
    limitacoes = ["Comparação textual conservadora; não realiza inferência semântica geral."]
    if not documento.url:
        limitacoes.append("URL de origem ausente; não é possível excluir o próprio publicador.")
    if parcial:
        limitacoes.append("Apenas as primeiras oito alegações foram selecionadas.")
    candidatos = [(u, "url_fornecida") for u in dict.fromkeys(evidence_urls)][:5]
    status_busca = "search_disabled" if not search else "search_not_configured"
    consultas = 0
    if search and not isinstance(provedores[0], SemBusca):
        status_busca = "completed"
        indice_provedor = 0
        consultaveis = [r for r in resultados if r["verificavel"]][:2]
        if len(consultaveis) == 1 and len(provedores) > 1:
            consultaveis *= 2
        for resultado in consultaveis:
            if consultas >= 2 or time.monotonic() >= prazo:
                break
            provedor = provedores[min(indice_provedor, len(provedores) - 1)]
            consultas += 1
            try:
                encontrados = _no_prazo(provedor.buscar, prazo, resultado["alegacao"][:250], prazo)
                candidatos.extend((e.url, e.provedor) for e in encontrados)
                if not encontrados:
                    indice_provedor += 1
            except Exception:
                status_busca = "retrieval_failed"
                limitacoes.append(f"Falha na consulta ao provedor {provedor.nome}.")
                indice_provedor += 1
                parcial = True
    if status_busca == "search_not_configured":
        limitacoes.append("Busca externa não configurada; scraping da notícia não é corroboração.")
    vistos, documentos = set(), []
    downloads = 0
    for url, provedor in candidatos:
        if url in vistos:
            continue
        if downloads >= 5 or time.monotonic() >= prazo:
            parcial = True
            limitacoes.append("Orçamento de downloads ou prazo global esgotado.")
            break
        vistos.add(url)
        downloads += 1
        try:
            outro = _no_prazo(extrator, prazo, url, prazo=prazo)
            fontes = fonte_url(outro.url)
            if documento.url and fontes == fonte_url(documento.url):
                limitacoes.append("Fonte do próprio publicador desconsiderada como independente.")
                continue
            if _copias(documento.corpo, outro.corpo) or any(
                fontes == fonte_url(d.url) or _copias(d.corpo, outro.corpo) for d, _ in documentos
            ):
                limitacoes.append("Cópia textual ou publicador repetido desconsiderado.")
                continue
            if outro.data_publicacao:
                try:
                    if date.fromisoformat(outro.data_publicacao[:10]) > referencia:
                        limitacoes.append(
                            "Evidência posterior à data de referência desconsiderada."
                        )
                        continue
                except ValueError:
                    limitacoes.append("Data da evidência não interpretável; não foi presumida.")
                    outro.data_publicacao = None
            documentos.append((outro, provedor))
        except Exception:
            parcial = True
            limitacoes.append("Uma URL de evidência não pôde ser extraída com segurança.")
    for resultado in resultados:
        if time.monotonic() >= prazo:
            parcial = True
            break
        if not resultado["verificavel"]:
            continue
        relacoes = set()
        for outro, provedor in documentos:
            sentencas, _ = dividir_alegacoes(outro.corpo, limite=100)
            for sentenca in sentencas:
                relacao, limites = comparar_alegacoes(resultado["alegacao"], sentenca["alegacao"])
                if relacao == "evidencia_insuficiente":
                    continue
                if not outro.data_publicacao:
                    limites.append("Data de publicação da evidência indisponível.")
                limites.append(
                    "Independência estimada por domínio e texto; "
                    "vínculos societários não auditados."
                )
                resultado["evidencias"].append(
                    {
                        "url": outro.url,
                        "trecho": sentenca["alegacao"],
                        "data": outro.data_publicacao,
                        "origem_data": outro.origem_data_publicacao,
                        "provedor": provedor,
                        "relacao": relacao,
                        "limitacoes": limites,
                        "publicador": fonte_url(outro.url),
                        "independente": True,
                    }
                )
                relacoes.add(relacao)
                break
        resultado["status"] = (
            "evidencias_conflitantes"
            if len(relacoes) > 1
            else (next(iter(relacoes)) if relacoes else "evidencia_insuficiente")
        )
    if time.monotonic() >= prazo:
        parcial = True
        limitacoes.append("Prazo global de verificação atingido.")
    return {
        "system_version": VERSAO_VERIFICACAO,
        "reference_date": referencia.isoformat(),
        "results": resultados,
        "summary": dict(Counter(r["status"] for r in resultados)),
        "partial": parcial,
        "limitations": list(dict.fromkeys(limitacoes)),
        "retrieval_status": status_busca,
        "latency_ms": round((time.monotonic() - inicio) * 1000, 2),
        "cache_hit": False,
        "buscas": consultas,
        "downloads": downloads,
    }
