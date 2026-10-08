export const CRIT = [
  { n: "Evidências", w: 39.8, d: "Dados, provas e referências consultáveis." },
  { n: "Qualidade das Fontes", w: 24.1, d: "Credibilidade e reputação das fontes." },
  { n: "Corroboração", w: 17.1, d: "Confirmação por fontes independentes." },
  { n: "Contexto", w: 11.8, d: "Sem recortes que alterem o sentido." },
  { n: "Atualidade", w: 7.2, d: "Recência e relevância temporal." },
];

// Protótipo: pontuações simuladas. Troque por chamada à sua API.
export function scoresFor(url) {
  if (url.includes("kaggle.com")) return [24, 19, 5, 33, 40];
  let h = 0;
  for (const c of url) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return CRIT.map((_, i) => Math.round(8 + ((h >>> (i * 5)) % 85)));
}

export const color = (v) =>
  v < 35 ? "var(--red)" : v < 65 ? "var(--amber)" : "var(--green)";

export const fmt = (n) => String(n).replace(".", ",");

export function analyze(url) {
  let host;
  try {
    host = new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return null;
  }
  const s = scoresFor(url);
  const total = Math.round(CRIT.reduce((a, c, i) => a + (c.w / 100) * s[i], 0));
  return { host, s, total, date: new Date().toLocaleDateString("pt-BR") };
}
