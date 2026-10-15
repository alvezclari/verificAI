const CRIT = [
  { key: "evidencias", n: "Evidências", w: 39.8, defaultD: "Dados, provas e referências consultáveis." },
  { key: "qualidade_fontes", n: "Qualidade das Fontes", w: 24.1, defaultD: "Credibilidade e reputação das fontes." },
  { key: "corroboracao", n: "Corroboração", w: 17.1, defaultD: "Confirmação por fontes independentes." },
  { key: "contexto", n: "Contexto", w: 11.8, defaultD: "Sem recortes que alterem o sentido." },
  { key: "atualidade", n: "Atualidade", w: 7.2, defaultD: "Recência e relevância temporal." }
];

// Identifica a URL da API (suporta tanto quando servido pelo Flask quanto abertura direta via arquivo)
const API_URL = (window.location.protocol === "http:" || window.location.protocol === "https:")
  ? "/api/analyze"
  : "http://localhost:5000/api/analyze";

const color = (v) => (v < 40 ? "var(--red)" : v < 70 ? "var(--amber)" : "var(--green)");
const $ = (id) => document.getElementById(id);
const fmt = (n) => String(n).replace(".", ",");

async function analyzeNews(url) {
  const submitBtn = $("form").querySelector("button");
  const originalBtnText = submitBtn.textContent;
  const errDiv = $("err");
  errDiv.textContent = "";

  try {
    submitBtn.disabled = true;
    submitBtn.textContent = "Analisando notícia...";
    submitBtn.style.opacity = "0.7";

    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    const data = await response.json();

    if (!response.ok || data.status === "error") {
      throw new Error(data.message || "Não foi possível analisar a notícia informada.");
    }

    renderResults(data);
  } catch (err) {
    errDiv.textContent = err.message || "Erro de conexão com o servidor. Verifique se o backend está ativo.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = originalBtnText;
    submitBtn.style.opacity = "1";
  }
}

function renderLocalExplanation(local) {
  const evidence = $("xai-evidence");
  const coverage = $("xai-coverage");
  const limits = $("xai-limits");
  evidence.replaceChildren();
  evidence.hidden = true;
  coverage.textContent = "";
  coverage.hidden = true;
  limits.textContent = "";
  limits.hidden = true;
  if (!local) return;

  const percentage = (value) => fmt((value * 100).toFixed(1)) + "%";
  [
    ["favorece_falsa", "Trechos que aumentaram a suspeita", "xai-fake"],
    ["favorece_confiavel", "Trechos que favoreceram a confiabilidade", "xai-reliable"],
  ].forEach(([direction, label, className]) => {
    const items = (local.trechos || []).filter((item) => item.direcao === direction).slice(0, 3);
    if (!items.length) return;
    const group = document.createElement("section");
    group.className = `xai-group ${className}`;
    const heading = document.createElement("h3");
    heading.textContent = label;
    group.append(heading);
    items.forEach((item) => {
      const figure = document.createElement("figure");
      const quote = document.createElement("blockquote");
      quote.textContent = item.texto;
      const effect = document.createElement("figcaption");
      effect.textContent = `${item.origem === "titulo" ? "Título" : "Corpo da notícia"} · Ao retirar este trecho, a estimativa da classe falsa passou de ${percentage(local.prob_fake_original)} para ${percentage(item.prob_fake_sem_trecho)}.`;
      figure.append(quote, effect);
      group.append(figure);
    });
    evidence.append(group);
  });
  evidence.hidden = !evidence.children.length;
  if (local.cobertura) {
    const info = local.cobertura;
    coverage.textContent = `Comparados ${info.trechos_avaliados} de ${info.trechos_identificados} trechos identificados.${info.parcial ? " A explicação cobre uma amostra; outros trechos também podem influenciar." : ""}`;
    coverage.hidden = false;
  }
  if (local.limitacoes?.length) {
    limits.textContent = local.limitacoes.join(" ");
    limits.hidden = false;
  }
}

function renderResults(data) {
  const host = data.metadata?.domain || new URL(data.url).hostname;
  const total = Number(data.confidence_score);
  const details = data.ml_details || {};
  $("technical-info").open = false;
  $("auxiliary-signals").open = false;
  const percentage = (value) => fmt((value * 100).toFixed(2)) + "%";
  const verdictColor = data.faixa
    ? { verdadeira: "var(--green)", inconclusiva: "var(--amber)", falsa: "var(--red)" }[data.faixa]
    : color(total);

  $("rows").replaceChildren();
  CRIT.forEach((criterion) => {
    const signal = data.criteria?.[criterion.key];
    const score = Math.round(signal?.score ?? 50);
    const row = document.createElement("div");
    row.className = "row";
    const top = document.createElement("div");
    top.className = "top";
    const label = document.createElement("b");
    label.textContent = criterion.key === "corroboracao" ? "Linguagem e sensacionalismo" : criterion.n;
    const value = document.createElement("b");
    value.textContent = score;
    top.append(label, value);
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.dataset.w = score;
    fill.style.background = color(score);
    bar.append(fill);
    const description = document.createElement("div");
    description.className = "desc";
    description.textContent = signal?.details || criterion.defaultD;
    row.append(top, bar, description);
    $("rows").append(row);
  });

  $("probabilities").textContent = details.probabilities
    ? `Classe falsa: ${percentage(details.probabilities.fake)} · Classe confiável: ${percentage(details.probabilities.confiavel)}`
    : `Score do modelo: ${fmt(total.toFixed(1))} de 100.`;
  $("components").replaceChildren();
  if (details.component_probabilities) {
    [["lexical", "TF-IDF + LogReg"], ["transformer", "mDeBERTa"]].forEach(([key, label]) => {
      const probability = details.component_probabilities[key];
      if (!probability) return;
      const line = document.createElement("p");
      const weight = details.weights?.[key];
      line.textContent = `${label}: P(falsa) ${percentage(probability.fake)}${weight == null ? "" : ` · peso ${percentage(weight)}`}`;
      $("components").append(line);
    });
  }
  const limits = data.thresholds;
  $("thresholds").textContent = limits
    ? `P(falsa) até ${percentage(limits.verdadeira_max)}: provavelmente verdadeira; acima disso até ${percentage(limits.inconclusiva_max)}: inconclusiva; acima: possível desinformação.`
    : "Resultado segundo o modelo selecionado na API.";
  $("model-info").textContent = [
    details.model_type,
    details.metodo_calibracao ? `Calibração: ${{sigmoid: "sigmoide", temperature_on_reserved_calibration_split: "temperatura do ensemble", sigmoid_on_reserved_calibration_split: "sigmoide do ensemble"}[details.metodo_calibracao] || details.metodo_calibracao}` : "",
    data.model_version ? `Versão: ${data.model_version}` : "",
    details.transformer ? `Transformer ativo · limite ${details.transformer.max_length} tokens` : "",
    details.transformer?.truncated ? `Texto truncado no ramo semântico: ${details.transformer.used_tokens} de ${details.transformer.input_tokens} tokens` : "",
    details.fator_texto_curto < 1 ? (details.transformer ? "Ramo léxico ajustado por texto curto." : "Confiança reduzida por texto curto.") : "",
  ].filter(Boolean).join(" · ");

  const evaluation = data.model_evaluation;
  $("metrics").replaceChildren();
  if (evaluation) {
    const test = evaluation.splits?.test;
    const calibration = evaluation.calibration_oof?.[details.metodo_calibracao];
    const metrics = [
      ["F1 macro (validação cruzada no treino)", evaluation.f1_macro_cv_train == null ? null : percentage(evaluation.f1_macro_cv_train)],
      ["Log loss (calibração OOF)", calibration?.log_loss?.toFixed(4)],
      ["Brier score (calibração OOF)", calibration?.brier?.toFixed(4)],
      ["Protocolo", evaluation.protocol?.startsWith("experimental") ? "Experimental, com avaliação reservada" : evaluation.protocol],
      ["F1 macro (teste do ensemble)", test?.ensemble ? percentage(test.ensemble.f1_macro) : null],
      ["F1 macro (teste do mDeBERTa)", test?.transformer ? percentage(test.transformer.f1_macro) : null],
      ["Log loss (teste do ensemble)", test?.ensemble?.log_loss?.toFixed(4)],
      ["Brier score (teste do ensemble)", test?.ensemble?.brier_score?.toFixed(4)],
    ];
    metrics.forEach(([label, value]) => {
      if (value == null) return;
      const line = document.createElement("p");
      line.textContent = `${label}: ${value}`;
      $("metrics").append(line);
    });
  } else {
    $("metrics").textContent = "Métricas não informadas pelo artefato selecionado.";
  }

  const explanation = details.explicacao;
  const band = data.faixa || {
    confiavel: "verdadeira", inconclusiva: "inconclusiva", desconfiavel: "falsa",
  }[data.verdict];
  const local = explanation?.local;
  $("unc").textContent = local?.resumo || {
    verdadeira: "O texto apresenta sinais favoráveis à confiabilidade. Use a estimativa como apoio à leitura e confira a origem e as informações da notícia.",
    inconclusiva: "Os sinais do texto não permitem uma conclusão clara sobre a confiabilidade. Busque outras fontes e mais contexto antes de compartilhar.",
    falsa: "O texto apresenta sinais que reduzem a estimativa de confiabilidade. Verifique as alegações em outras fontes antes de compartilhar.",
  }[band] || "A estimativa resume os sinais encontrados no texto. Confira a origem e compare as informações com outras fontes antes de compartilhar.";
  renderLocalExplanation(local);
  $("explanation-details").textContent = [data.summary,
    explanation?.scope === "lexical_component_only" ? "As explicações lexicais se referem apenas ao ramo TF-IDF + LogReg." : "",
    ...(explanation?.limitacoes || [])].filter(Boolean).join(" ");
  const verdict = $("verdict");
  verdict.textContent = data.verdict_label;
  verdict.style.color = verdictColor;
  $("meta").textContent = [host, data.metadata?.author].filter(Boolean).join(" · ");
  $("home").style.display = "none";
  $("result").style.display = "block";
  $("score").textContent = fmt(total.toFixed(1));
  window.scrollTo(0, 0);
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll(".bar div").forEach((bar) => { bar.style.width = bar.dataset.w + "%"; });
    $("arc").setAttribute("stroke-dasharray", `${total} 100`);
  }));
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  let u = $("url").value.trim();
  if (u && !/^https?:\/\//i.test(u)) u = "https://" + u;

  if (!u) {
    $("err").textContent = "Cole um link completo, como https://exemplo.com.br/noticia";
    return;
  }

  analyzeNews(u);
});

const goToHome = () => {
  $("result").style.display = "none";
  $("home").style.display = "block";
  $("url").value = "";
  $("err").textContent = "";
  $("url").focus();
  window.scrollTo(0, 0);
};

$("back").onclick = goToHome;

// Torna o nome/logo VerificAI no canto superior esquerdo clicável para voltar ao início
const brandLogo = $("brand-logo") || document.querySelector(".brand");
if (brandLogo) {
  brandLogo.onclick = goToHome;
  brandLogo.onkeydown = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      goToHome();
    }
  };
}
