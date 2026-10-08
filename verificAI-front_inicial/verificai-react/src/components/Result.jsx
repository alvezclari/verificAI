import { useEffect, useState } from "react";
import { CRIT, color, fmt } from "../data.js";

export default function Result({ data, onBack }) {
  const { host, s, total, date } = data;
  const [on, setOn] = useState(false);

  useEffect(() => {
    window.scrollTo(0, 0);
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setOn(true)));
    return () => cancelAnimationFrame(id);
  }, []);

  const hi = s.indexOf(Math.max(...s));
  const lo = s.indexOf(Math.min(...s));
  const verdict =
    total < 40 ? "Baixa confiabilidade estimada"
    : total < 70 ? "Confiabilidade moderada estimada"
    : "Alta confiabilidade estimada";

  return (
    <section>
      <div className="grid">
        <div className="card gauge">
          <div className="sub" style={{ textAlign: "left", letterSpacing: ".08em" }}>INDICADOR GERAL</div>
          <svg viewBox="0 0 340 200" aria-hidden="true">
            <path d="M40 180 A130 130 0 0 1 300 180" fill="none" stroke="#1c2c4b" strokeWidth="30" strokeLinecap="round" />
            <defs>
              <linearGradient id="g">
                <stop offset="0" stopColor="#e5655b" />
                <stop offset="1" stopColor="#ef9a5a" />
              </linearGradient>
            </defs>
            <path
              id="arc"
              d="M40 180 A130 130 0 0 1 300 180"
              fill="none"
              stroke="url(#g)"
              strokeWidth="30"
              strokeLinecap="round"
              pathLength="100"
              strokeDasharray={`${on ? total : 0} 100`}
              style={{ transition: "stroke-dasharray 1s cubic-bezier(.2,.8,.2,1)" }}
            />
          </svg>
          <div className="num">{total}</div>
          <div className="est">de 100 (estimativa)</div>
          <p className="verdict" style={{ color: color(total) }}>{verdict}</p>
          <div className="meta">{host} · analisado em {date}</div>
        </div>

        <div className="card crit">
          <h2>Análise por critério</h2>
          <div className="sub">Pesos definidos pelo método AHP (CR = 2,49%)</div>
          {CRIT.map((c, i) => (
            <div className="row" key={c.n}>
              <div className="top">
                <span><b>{c.n}</b> <i>· peso {fmt(c.w)}%</i></span>
                <b>{s[i]}</b>
              </div>
              <div className="bar">
                <div style={{ width: on ? s[i] + "%" : 0, background: color(s[i]) }} />
              </div>
              <div className="desc">{c.d}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="three">
        <div className="card">
          <h3 style={{ color: "var(--green)" }}>O que sustenta</h3>
          <p>Ponto mais forte: <b>{CRIT[hi].n}</b> ({s[hi]}). {CRIT[hi].d}</p>
        </div>
        <div className="card">
          <h3 style={{ color: "var(--amber)" }}>O que enfraquece</h3>
          <p>Ponto mais fraco: <b>{CRIT[lo].n}</b> ({s[lo]}). Vale investigar este aspecto.</p>
        </div>
        <div className="card">
          <h3 style={{ color: "var(--teal)" }}>O que não foi confirmado</h3>
          <p>
            {s[2] < 40
              ? "Poucas fontes independentes confirmaram o fato até agora."
              : "Verifique por conta própria os dados citados na matéria."}
          </p>
        </div>
      </div>

      <button className="back" type="button" onClick={onBack}>Verificar outro link</button>
    </section>
  );
}
