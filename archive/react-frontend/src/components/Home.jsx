import { useState } from "react";
import { analyze } from "../data.js";

export default function Home({ onResult }) {
  const [url, setUrl] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    let u = url.trim();
    if (u && !/^https?:\/\//i.test(u)) u = "https://" + u;

    if (!u) {
      setErr("Cole um link completo, como https://exemplo.com.br/noticia");
      return;
    }

    try {
      setLoading(true);
      setErr("");
      const r = await analyze(u);
      onResult(r);
    } catch (error) {
      setErr(error.message || "Erro ao conectar com o backend.");
    } finally {
      setLoading(false);
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
          disabled={loading}
        />
        <button type="submit" disabled={loading} style={{ opacity: loading ? 0.7 : 1 }}>
          {loading ? "Analisando..." : "Verificar"}
        </button>
      </form>
      <div className="err" role="alert">{err}</div>
      <p className="note">A VerificAI organiza sinais e evidências — a decisão final é sempre do leitor.</p>
    </section>
  );
}
