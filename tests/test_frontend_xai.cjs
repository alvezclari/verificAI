const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

class Element {
  constructor(tag = "div") { this.tag = tag; this.children = []; this.textContent = ""; this.hidden = false; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener() {}
  set innerHTML(_) { throw new Error("Texto de notícia não deve ser interpretado como HTML."); }
}

function page() {
  const elements = new Map();
  const context = vm.createContext({
    window: { location: { protocol: "http:" } },
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
      },
      createElement: (tag) => new Element(tag),
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../frontend/app.js"), "utf8"), context);
  return { elements, render: (value) => context.renderLocalExplanation(value) };
}

test("mostra os dois sentidos da explicação e mantém o trecho como texto", () => {
  const { elements, render } = page();
  render({
    prob_fake_original: 0.82,
    trechos: [
      { direcao: "favorece_falsa", texto: '<img src=x onerror="alert(1)">', origem: "titulo", prob_fake_sem_trecho: 0.64 },
      { direcao: "favorece_confiavel", texto: "O documento apresenta os dados.", origem: "corpo", prob_fake_sem_trecho: 0.9 },
    ],
    cobertura: { trechos_avaliados: 8, trechos_identificados: 20, parcial: true },
    limitacoes: ["Os efeitos não são somáveis."],
  });
  const evidence = elements.get("xai-evidence");
  assert.equal(evidence.hidden, false);
  assert.equal(evidence.children.length, 2);
  assert.equal(evidence.children[0].children[1].children[0].textContent, '<img src=x onerror="alert(1)">');
  assert.match(evidence.children[0].children[1].children[1].textContent, /82,0% para 64,0%/);
  assert.match(elements.get("xai-coverage").textContent, /8 de 20/);
  assert.match(elements.get("xai-coverage").textContent, /amostra/);
});

test("uma resposta sem XAI ou com falha apaga trechos da notícia anterior", () => {
  const { elements, render } = page();
  render({ prob_fake_original: 0.8, trechos: [{ direcao: "favorece_falsa", texto: "Trecho antigo", prob_fake_sem_trecho: 0.6 }] });
  render(undefined);
  assert.equal(elements.get("xai-evidence").hidden, true);
  assert.equal(elements.get("xai-evidence").children.length, 0);
  assert.equal(elements.get("xai-coverage").hidden, true);
  assert.equal(elements.get("xai-limits").hidden, true);
  render({ status: "unavailable", trechos: [] });
  assert.equal(elements.get("xai-evidence").children.length, 0);
});
