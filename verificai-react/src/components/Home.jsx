import { useState } from "react";
import { analyze } from "../data.js";

export default function Home({ onResult }) {
  const [url, setUrl] = useState("");
  const [err, setErr] = useState("");

  const submit = (e) => {
    e.preventDefault();
    let u = url.trim();
    if (u && !/^https?:\/\//i.test(u)) u = "https://" + u;
    const r = u ? analyze(u) : null;
    if (!r) setErr("Cole um link completo, como https://exemplo.com.br/noticia");
    else {
      setErr("");
      onResult(r);
    }
  };

  return (
    <section className="hero">
      <h1>Antes de compartilhar,<br /><span>observe os sinais.</span></h1>
      <p>Cole o link de uma notícia. Mostramos um indicador de confiabilidade explicado por 5 critérios — a conclusão final é sua.</p>
      <form onSubmit={submit} noValidate>
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://exemplo.com.br/noticia"
          aria-label="Link da notícia"
          autoComplete="off"
        />
        <button type="submit">Verificar</button>
      </form>
      <div className="err" role="alert">{err}</div>
      <p className="note">A VerificAI organiza sinais e evidências — a decisão final é sempre do leitor.</p>
    </section>
  );
}
