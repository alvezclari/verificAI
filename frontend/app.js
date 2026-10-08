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

function renderResults(data) {
  const host = data.metadata?.domain || new URL(data.url).hostname.replace(/^www\./, "");
  const total = Math.round(data.confidence_score);

  // Extrai pontuações e descrições dos 5 critérios a partir da resposta da API
  const s = CRIT.map((c) => Math.round(data.criteria?.[c.key]?.score ?? 50));
  const descs = CRIT.map((c) => data.criteria?.[c.key]?.details || c.defaultD);

  // Renderiza as barras dos critérios
  $("rows").innerHTML = CRIT.map((c, i) => `
    <div class="row">
      <div class="top">
        <span><b>${c.n}</b> <i>· peso ${fmt(c.w)}%</i></span>
        <b>${s[i]}</b>
      </div>
      <div class="bar">
        <div data-w="${s[i]}" style="background:${color(s[i])}"></div>
      </div>
      <div class="desc">${descs[i]}</div>
    </div>
  `).join("");

  // Ponto mais forte e mais fraco
  const hi = s.indexOf(Math.max(...s));
  const lo = s.indexOf(Math.min(...s));

  $("best").innerHTML = `Ponto mais forte: <b>${CRIT[hi].n}</b> (${s[hi]}). ${descs[hi]}`;
  $("worst").innerHTML = `Ponto mais fraco: <b>${CRIT[lo].n}</b> (${s[lo]}). ${descs[lo]}`;

  // Síntese / Terceiro card
  if (data.summary) {
    $("unc").textContent = data.summary;
  } else {
    $("unc").textContent = s[2] < 40
      ? "Poucas fontes independentes confirmaram o fato até agora."
      : "Verifique por conta própria os dados citados na matéria.";
  }

  // Veredito geral
  const v = $("verdict");
  v.textContent = data.verdict_label || (total >= 50 ? "Confiável" : "Desconfiável / Suspeita de Fake News");
  v.style.color = color(total);

  // Metadados
  const authorInfo = data.metadata?.author ? ` · ${data.metadata.author}` : "";
  const mlInfo = data.ml_details ? ` · Modelo: ${data.ml_details.model_type}` : "";
  $("meta").textContent = `${host}${authorInfo}${mlInfo}`;

  // Transição de tela
  $("home").style.display = "none";
  $("result").style.display = "block";
  window.scrollTo(0, 0);

  // Atualização do score e animação
  $("score").textContent = total;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll(".bar div").forEach((b) => (b.style.width = b.dataset.w + "%"));
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
