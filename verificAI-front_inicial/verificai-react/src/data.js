export const CRIT = [
  { key: "evidencias", n: "Evidências", w: 39.8, defaultD: "Dados, provas e referências consultáveis." },
  { key: "qualidade_fontes", n: "Qualidade das Fontes", w: 24.1, defaultD: "Credibilidade e reputação das fontes." },
  { key: "corroboracao", n: "Corroboração", w: 17.1, defaultD: "Confirmação por fontes independentes." },
  { key: "contexto", n: "Contexto", w: 11.8, defaultD: "Sem recortes que alterem o sentido." },
  { key: "atualidade", n: "Atualidade", w: 7.2, defaultD: "Recência e relevância temporal." },
];

export const color = (v) =>
  v < 40 ? "var(--red)" : v < 70 ? "var(--amber)" : "var(--green)";

export const fmt = (n) => String(n).replace(".", ",");

const API_URL = "http://localhost:5000/api/analyze";

export async function analyze(url) {
  let host;
  try {
    host = new URL(url).hostname.replace(/^www\./, "");
  } catch {
    throw new Error("Formato de URL inválido.");
  }

  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url })
  });

  const data = await response.json();

  if (!response.ok || data.status === "error") {
    throw new Error(data.message || "Não foi possível analisar a notícia informada.");
  }

  const total = Math.round(data.confidence_score);
  const s = CRIT.map((c) => Math.round(data.criteria?.[c.key]?.score ?? 50));
  const descs = CRIT.map((c) => data.criteria?.[c.key]?.details || c.defaultD);

  return {
    host: data.metadata?.domain || host,
    s,
    descs,
    total,
    verdictLabel: data.verdict_label,
    summary: data.summary,
    date: data.metadata?.publish_date || new Date().toLocaleDateString("pt-BR"),
    author: data.metadata?.author,
    mlDetails: data.ml_details
  };
}
