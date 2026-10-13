"""Carga, auditoria e separação por publicador, sem consultas à rede."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from ml.preprocessing import preparar_texto

ROTULOS = {0: "falsa", 1: "verdadeira"}
# Usa exclusivamente a lista de sufixos distribuída no pacote, sem download/cache externo.
_DOMINIO = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
COLUNAS = ["Titulo", "Noticia", "URL", "Classe", "Data"]


class ProtocoloInviavel(ValueError):
    """Os dados não permitem cumprir as separações exigidas."""


class _Uniao:
    def __init__(self, n):
        self.pais = list(range(n))

    def raiz(self, i):
        while self.pais[i] != i:
            self.pais[i] = self.pais[self.pais[i]]
            i = self.pais[i]
        return i

    def unir(self, i, j):
        a, b = self.raiz(i), self.raiz(j)
        self.pais[max(a, b)] = min(a, b)


def hash_arquivo(caminho: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as entrada:
        for bloco in iter(lambda: entrada.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def salvar_json(caminho: str | Path, valor) -> None:
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(valor, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def normalizar_url(valor: str) -> str:
    url = urlsplit(valor.strip())
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError(f"URL inválida no dataset: {valor!r}")
    consulta = [
        (k, v)
        for k, v in parse_qsl(url.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}
    ]
    return urlunsplit(
        (url.scheme.lower(), url.netloc.lower(), url.path or "/", urlencode(sorted(consulta)), "")
    )


def fonte_url(valor: str) -> str:
    host = urlsplit(valor).hostname
    extraido = _DOMINIO(host)
    return extraido.top_domain_under_public_suffix or host


def carregar_dataset(caminho: str | Path) -> tuple[pd.DataFrame, dict]:
    """Lê somente colunas de origem; nunca usa texto_limpo/texto_processado."""
    caminho = Path(caminho)
    if caminho.suffix.lower() == ".xlsx":
        dados = pd.read_excel(caminho, dtype=str).fillna("")
    elif caminho.suffix.lower() == ".csv":
        dados = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    else:
        raise ValueError("Formato não suportado: forneça CSV ou XLSX.")
    ausentes = sorted(set(COLUNAS) - set(dados.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {ausentes}")
    # A planilha pode conter linhas totalmente vazias, sem registro de notícia.
    vazias = dados[COLUNAS].apply(lambda c: c.str.strip()).eq("").all(axis=1)
    dados = dados.loc[~vazias, COLUNAS].copy()
    if not dados.Classe.isin(["0", "1"]).all():
        invalidos = dados.loc[~dados.Classe.isin(["0", "1"]), "Classe"].value_counts().to_dict()
        raise ValueError(f"Rótulos inválidos; esperado 0=falsa e 1=verdadeira: {invalidos}")
    if dados.Noticia.str.strip().eq("").any():
        raise ValueError("Existem notícias sem corpo; revise os dados antes do treino.")
    sha = hash_arquivo(caminho)
    dados["id"] = [hashlib.sha256(f"{sha}:{i}".encode()).hexdigest() for i in dados.index]
    dados["label"] = dados.Classe.astype(int)
    dados["url_normalizada"] = dados.URL.map(normalizar_url)
    dados["fonte"] = dados.url_normalizada.map(fonte_url)
    dados["texto"] = [preparar_texto(t, c) for t, c in zip(dados.Titulo, dados.Noticia)]
    dados["hash_texto"] = dados.texto.map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    # Normaliza apenas formatos editoriais explícitos do próprio campo Data.
    # A atualização posterior não substitui a primeira data de publicação.
    datas_texto = (
        dados.Data.str.replace(r"(?i)atualizad[ao].*$", "", regex=True)
        .str.replace(r"(?i)^\s*publicado\s+em\s*", "", regex=True)
        .str.replace("às", " ", regex=False)
        .str.replace(r"(\d{1,2})h(\d{2})", r"\1:\2", regex=True)
        .str.strip()
    )
    datas = pd.to_datetime(datas_texto, format="mixed", dayfirst=True, errors="coerce", utc=True)
    dados["data"] = datas
    fontes = pd.crosstab(dados.fonte, dados.label).reindex(columns=[0, 1], fill_value=0)
    auditoria = {
        "dataset": str(caminho),
        "sha256": sha,
        "linhas": len(dados),
        "linhas_totalmente_vazias_removidas": int(vazias.sum()),
        "rotulos": {str(k): v for k, v in ROTULOS.items()},
        "balanceamento": {str(k): int(v) for k, v in dados.label.value_counts().items()},
        "por_publicador": {
            fonte: {"falsas": int(row[0]), "verdadeiras": int(row[1])}
            for fonte, row in fontes.iterrows()
        },
        "datas_validas": int(datas.notna().sum()),
        "datas_ausentes": int(dados.Data.str.strip().eq("").sum()),
        "datas_invalidas_nao_vazias": int((datas.isna() & dados.Data.str.strip().ne("")).sum()),
        "data_minima": str(datas.min()) if datas.notna().any() else None,
        "data_maxima": str(datas.max()) if datas.notna().any() else None,
        "politica_datas": (
            "Campo Data, dia/mês/ano; remove prefixo Publicado em e sufixo Atualizada; "
            "normaliza às e hhHmm. Sem fuso explícito, UTC é convenção de ordenação, "
            "não horário de publicação confirmado. Não corrige anos malformados."
        ),
        "titulos_ausentes": int(dados.Titulo.str.strip().eq("").sum()),
        "corpos_com_menos_de_15_palavras": int(dados.Noticia.str.split().str.len().lt(15).sum()),
        "duplicatas_url": int(dados.url_normalizada.duplicated().sum()),
        "duplicatas_texto": int(dados.hash_texto.duplicated().sum()),
        "conflitos_url": int(dados.groupby("url_normalizada").label.nunique().gt(1).sum()),
        "conflitos_texto": int(dados.groupby("hash_texto").label.nunique().gt(1).sum()),
        "suporte_publicadores_por_classe_antes_deduplicacao": {
            str(k): int(dados.loc[dados.label.eq(k), "fonte"].nunique()) for k in (0, 1)
        },
        "limitacoes": [
            "Corpos atuais já transformados; não existe corpo original disponível.",
            "Amostras incluem reportagens de checagem rotuladas pela alegação discutida.",
            "Mapeamento 0=falsa, 1=verdadeira confirmado pelo responsável pelos dados.",
            "Não há URLs com verdade factual conhecida para teste externo.",
            "Métricas desse corpus não certificam calibração de artigos atuais extraídos.",
            "Domínio registrável aproxima publicador; não identifica todo grupo empresarial.",
        ],
    }
    auditoria["viabilidade_fontes_antes_deduplicacao"] = validar_viabilidade(
        dados.assign(grupo=dados.fonte)
    )
    return dados.reset_index(drop=True), auditoria


def exportar_pendencias(dados: pd.DataFrame, destino: Path) -> None:
    """Gera fichas de revisão sem mudar rótulos ou excluir notícias automaticamente."""
    destino.mkdir(parents=True, exist_ok=True)
    conflitantes = set()
    for coluna in ("url_normalizada", "hash_texto"):
        totais = dados.groupby(coluna).label.nunique()
        chaves = totais.index[totais.gt(1)]
        conflitantes.update(dados.loc[dados[coluna].isin(chaves), "id"])
    ficha = dados.loc[
        dados.id.isin(conflitantes), ["id", "Titulo", "URL", "label", "fonte", "hash_texto"]
    ].copy()
    revisao = destino / "conflitos_rotulos.csv"
    colunas_revisao = ("decisao", "justificativa", "revisor")
    anterior = None
    if revisao.exists():
        anterior = pd.read_csv(revisao, dtype=str, keep_default_na=False).set_index("id")
        if not anterior.index.is_unique:
            raise ValueError("A ficha de revisão existente contém IDs duplicados.")
    for coluna in colunas_revisao:
        if anterior is not None and coluna in anterior:
            ficha[coluna] = ficha.id.map(anterior[coluna]).fillna("")
        else:
            ficha[coluna] = ""
    ficha.to_csv(revisao, index=False)
    datas = dados.loc[dados.data.isna(), ["id", "URL", "Data"]].copy()
    datas["motivo"] = np.where(datas.Data.str.strip().eq(""), "ausente", "formato_nao_reconhecido")
    datas.to_csv(destino / "datas_pendentes.csv", index=False)


def excluir_conflitos_url(dados: pd.DataFrame, destino: Path) -> tuple[pd.DataFrame, dict]:
    """Aplica a exclusão autorizada, sem editar os arquivos de origem ou os rótulos."""
    contagens = dados.groupby("url_normalizada").label.nunique()
    urls = contagens.index[contagens.gt(1)]
    mascara = dados.url_normalizada.isin(urls)
    removidos = dados.loc[mascara, ["id", "URL", "url_normalizada", "label", "hash_texto"]].copy()
    removidos["motivo"] = "url_com_rotulos_conflitantes"
    removidos["politica"] = "exclusao_experimental_autorizada_pelo_usuario"
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / "exclusoes_conflitos.csv"
    removidos.to_csv(arquivo, index=False)
    return dados.loc[~mascara].reset_index(drop=True), {
        "registros": int(mascara.sum()),
        "urls": len(urls),
        "arquivo": arquivo.name,
        "sha256": hash_arquivo(arquivo),
        "rotulos_removidos": {str(k): int(v) for k, v in removidos.label.value_counts().items()},
    }


def deduplicar(dados: pd.DataFrame, agrupar_por_fonte: bool = True) -> pd.DataFrame:
    """Une relações por URL/texto antes de remover duplicatas, inclusive transitivas."""
    dados = dados.reset_index(drop=True).copy()
    uniao = _Uniao(len(dados))
    colunas_grupo = (
        ("url_normalizada", "hash_texto", "fonte")
        if agrupar_por_fonte
        else ("url_normalizada", "hash_texto")
    )
    for coluna in colunas_grupo:
        primeiros = {}
        for i, chave in enumerate(dados[coluna]):
            if chave in primeiros:
                uniao.unir(i, primeiros[chave])
            else:
                primeiros[chave] = i
    for coluna in ("url_normalizada", "hash_texto"):
        if dados.groupby(coluna).label.nunique().gt(1).any():
            raise ProtocoloInviavel(
                f"Rótulos conflitantes para a mesma {coluna}; revisão necessária."
            )
    dados["grupo"] = [f"grupo_{uniao.raiz(i)}" for i in range(len(dados))]
    # Não deixa a remoção de uma linha esconder a ligação entre dois publicadores.
    return (
        dados.drop_duplicates("url_normalizada")
        .drop_duplicates("hash_texto")
        .reset_index(drop=True)
    )


def validar_viabilidade(dados: pd.DataFrame) -> dict:
    """Condição necessária: dois grupos por classe no treino, um em cada reserva."""
    suporte = {str(k): int(dados.loc[dados.label.eq(k), "grupo"].nunique()) for k in (0, 1)}
    motivos = []
    for classe, n in suporte.items():
        if n < 4:
            motivos.append(
                f"Classe {ROTULOS[int(classe)]}: apenas {n} grupos de separação; são necessários "
                "ao menos 4 para treino com validação cruzada agrupada, calibração e teste "
                "contendo ambas as classes. A reserva temporal pode exigir grupos adicionais."
            )
    return {"viavel": not motivos, "grupos_por_classe": suporte, "motivos": motivos}


def agrupar_quase_duplicatas(dados: pd.DataFrame, similaridade: float = 0.92) -> tuple:
    """Liga publicadores com cópias semelhantes; não aprende vocabulário ou rótulos."""
    if not 0 < similaridade <= 1:
        raise ValueError("Similaridade deve estar em (0, 1].")
    dados = dados.reset_index(drop=True).copy()
    uniao = _Uniao(len(dados))
    primeiros = {}
    for i, grupo in enumerate(dados.grupo):
        uniao.unir(i, primeiros.setdefault(grupo, i))
    matriz = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=2**18,
        alternate_sign=False,
        lowercase=False,
    ).transform(dados.texto)
    vizinhos = NearestNeighbors(metric="cosine", radius=1 - similaridade, n_jobs=1).fit(matriz)
    pares = 0
    for inicio in range(0, len(dados), 128):
        indices = vizinhos.radius_neighbors(matriz[inicio : inicio + 128], return_distance=False)
        for deslocamento, relacionados in enumerate(indices):
            i = inicio + deslocamento
            for j in relacionados:
                if j > i:
                    uniao.unir(i, int(j))
                    pares += 1
    dados["grupo"] = [f"grupo_{uniao.raiz(i)}" for i in range(len(dados))]
    return dados, {
        "similaridade_minima": similaridade,
        "pares": pares,
        "grupos_resultantes": int(dados.grupo.nunique()),
    }


def dobras_agrupadas(dados: pd.DataFrame, n_splits: int = 3) -> list:
    """Valida cada dobra antes de entregar índices ao treinamento."""
    if dados.grupo.nunique() < n_splits:
        raise ProtocoloInviavel("Grupos insuficientes para validação cruzada.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    dobras = list(cv.split(dados.texto, dados.label, dados.grupo))
    for treino, validacao in dobras:
        if set(dados.iloc[treino].label) != {0, 1} or set(dados.iloc[validacao].label) != {0, 1}:
            raise ProtocoloInviavel("Uma dobra agrupada ficou sem as duas classes.")
        if set(dados.iloc[treino].grupo) & set(dados.iloc[validacao].grupo):
            raise AssertionError("Vazamento de grupos na validação cruzada.")
    return dobras


@dataclass
class Particoes:
    treino: pd.DataFrame
    calibracao: pd.DataFrame
    teste: pd.DataFrame
    temporal: pd.DataFrame
    excluidos_temporal: pd.DataFrame
    corte_temporal: str | None

    def salvar(self, caminho: Path) -> None:
        conteudo = {
            nome: getattr(self, nome).id.tolist()
            for nome in ("treino", "calibracao", "teste", "temporal", "excluidos_temporal")
        }
        conteudo["corte_temporal"] = self.corte_temporal
        salvar_json(caminho, conteudo)


def particionar(dados: pd.DataFrame, seed: int = 42, fracao_temporal: float = 0.15) -> Particoes:
    """Reserva o período recente e purga seus grupos do desenvolvimento e teste comum."""
    if not 0 <= fracao_temporal < 1:
        raise ValueError("Fração temporal deve estar em [0, 1).")
    diagnostico = validar_viabilidade(dados)
    if not diagnostico["viavel"]:
        raise ProtocoloInviavel(" ".join(diagnostico["motivos"]))
    temporal = dados.iloc[:0].copy()
    excluidos = temporal.copy()
    corte = None
    desenvolvimento = dados
    if fracao_temporal and dados.data.notna().any():
        corte_data = dados.data.dropna().quantile(1 - fracao_temporal)
        temporal = dados.loc[dados.data.ge(corte_data)].copy()
        grupos = set(temporal.grupo)
        # Datas desconhecidas não podem ser consideradas anteriores ao corte.
        mascara = dados.grupo.isin(grupos) | dados.data.isna()
        excluidos = dados.loc[mascara & ~dados.id.isin(temporal.id)].copy()
        desenvolvimento = dados.loc[~mascara].copy()
        corte = corte_data.isoformat()
        if set(temporal.label) != {0, 1}:
            raise ProtocoloInviavel("Reserva temporal não contém ambas as classes.")
        viabilidade = validar_viabilidade(desenvolvimento)
        if not viabilidade["viavel"]:
            raise ProtocoloInviavel(
                "Após purgar grupos do período recente: " + " ".join(viabilidade["motivos"])
            )
    melhor = None
    perda_melhor = np.inf
    externo = GroupShuffleSplit(n_splits=200, test_size=0.30, random_state=seed)
    for indices_treino, indices_reserva in externo.split(
        desenvolvimento, groups=desenvolvimento.grupo
    ):
        treino = desenvolvimento.iloc[indices_treino]
        reserva = desenvolvimento.iloc[indices_reserva]
        interno = GroupShuffleSplit(n_splits=5, test_size=0.50, random_state=seed)
        for a, b in interno.split(reserva, groups=reserva.grupo):
            cal, teste = reserva.iloc[a], reserva.iloc[b]
            if any(set(p.label) != {0, 1} for p in (treino, cal, teste)):
                continue
            perda = sum(
                abs(len(p) / len(desenvolvimento) - alvo)
                + abs(float(p.label.mean()) - float(desenvolvimento.label.mean()))
                for p, alvo in zip((treino, cal, teste), (0.70, 0.15, 0.15))
            )
            if perda >= perda_melhor:
                continue
            try:
                dobras_agrupadas(treino)
                dobras_agrupadas(cal)
            except ProtocoloInviavel:
                continue
            melhor = (treino.copy(), cal.copy(), teste.copy())
            perda_melhor = perda
    if melhor is None:
        raise ProtocoloInviavel(
            "Não foi encontrada partição agrupada com ambas as classes e CV válida."
        )
    partes = Particoes(*melhor, temporal, excluidos, corte)
    conjuntos = [set(p.grupo) for p in (*melhor, temporal)]
    if any(a & b for i, a in enumerate(conjuntos) for b in conjuntos[i + 1 :]):
        raise AssertionError("Grupos compartilhados entre partições.")
    return partes
