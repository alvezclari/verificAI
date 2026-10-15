# Explicação local da classificação

A aba **Como interpretar** apresenta um resumo específico da notícia e trechos
que influenciaram a probabilidade final. A primeira implementação usa **remoção
de frases** (`sentence_deletion`), sem LLM, novos pesos ou retreinamento.
Não é SHAP: os efeitos individuais não seguem uma decomposição aditiva.

## O que é medido

1. A previsão original é calculada normalmente.
2. O título e as frases do corpo com pelo menos quatro palavras são candidatos;
   o título é elegível quando tem pelo menos três palavras e existe corpo.
3. Por padrão, comparam-se até oito candidatos. O título tem prioridade; as
   frases restantes são amostradas de forma uniforme ao longo do corpo.
4. Para cada candidato, remove-se apenas seu intervalo do título ou corpo.
5. A variante passa pelo mesmo pré-processamento, modelos, ajuste de texto curto,
   pesos e calibração da previsão original.

O efeito é `100 × (P(falsa) original − P(falsa) sem o trecho)`, em **pontos
percentuais**. Um efeito positivo significa que retirar o trecho reduziu a
estimativa da classe falsa; um negativo indica o sentido oposto. Efeitos com
módulo inferior a 0,5 ponto percentual não são destacados.

O ensemble calcula as variantes em lotes de dois, limitando o uso adicional de
memória. A explicação usa a referência do modelo capturada no início da análise,
inclusive quando outra requisição recarrega o serviço durante a inferência.

## Resumo e interface

O resumo usa apenas probabilidades medidas e trechos efetivamente comparados.
A interface mostra até três exemplos em cada sentido e informa quantos candidatos
foram avaliados. Quando não há efeito expressivo ou texto suficiente, isso fica
explícito; não se inventa uma justificativa com palavras suspeitas ou heurísticas.

Os cinco sinais auxiliares continuam informativos e não entram na pontuação do
ensemble. As contribuições de palavras existentes continuam se referindo somente
ao ramo léxico; a nova explicação é separada em `ml_details.explicacao.local`.

## API

`POST /api/analyze` mantém a probabilidade, o score, as faixas e os campos anteriores.
Quando habilitada no modo calibrado ou no ensemble supervisionado v2, a resposta
acrescenta `ml_details.explicacao.local`:

| Campo | Significado |
|---|---|
| `metodo`, `scope` | `sentence_deletion`, `final_prediction` |
| `status` | `ok`, `no_effect`, `insufficient_text` ou `unavailable` |
| `resumo` | Texto determinístico construído com as comparações medidas |
| `prob_fake_original` | A mesma probabilidade usada na classificação exibida |
| `trechos` | Até seis destaques, com texto, origem, offsets, probabilidade sem o trecho, efeito e direção |
| `avaliacoes` | Todas as comparações realizadas, inclusive efeitos pequenos |
| `cobertura` | Contagens, seleção parcial e truncamento semântico da entrada original |
| `limitacoes` | Limites da interpretação e da cobertura |
| `tempo_ms` | Tempo adicional gasto pelo explicador |

Os offsets são índices de caracteres Unicode (code points), com fim exclusivo,
no título ou corpo original extraído. A extração não inclui o conteúdo completo
na resposta; ela apresenta os metadados e os trechos retornados, sem persistência
automática do conteúdo.

Uma falha no explicador produz `status=unavailable`, sem substituir ou alterar
a classificação válida. Quando desabilitado, o campo `local` é omitido.
O frontend mantém compatibilidade com respostas anteriores e com o modo legado.

## Configuração e validação

| Variável | Padrão | Uso |
|---|---|---|
| `XAI_ENABLED` | `true` | Habilita a comparação por trechos |
| `XAI_MAX_EVALUATIONS` | `8` | Número de variantes; limitado entre 1 e 16 |

As variáveis precisam estar no ambiente antes de iniciar `run.py`; `.env` não é
carregado automaticamente. Use menos avaliações para reduzir a latência em CPU.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-classificador/bin/python -m pytest -q
node --test tests/test_frontend_xai.cjs
```

Os testes verificam sinais e probabilidades dos efeitos, offsets, limites de
inferência, falhas do explicador, ausência de motivos inventados, compatibilidade
com o motor original e com o ensemble calibrado, além da renderização de texto
não confiável e da limpeza dos resultados entre notícias.

Na validação local de 15/10/2026, o bundle real em CPU comparou oito de treze
candidatos de um artigo **sintético**, com cerca de 535 ms adicionais no explicador.
A probabilidade e a faixa foram idênticas às da execução sem explicação. Esse
resultado valida a execução técnica; não mede acurácia factual, qualidade humana
das justificativas nem estabelece uma garantia de latência para outros textos
ou equipamentos. O teste de integração por HTML controlado também passou.

## Limitações

- Remover frases muda contexto, comprimento e distribuição da entrada. Os efeitos
  são comparações locais, não causas de falsidade nem importâncias somáveis.
- A segmentação é heurística e reconhece abreviações comuns; não substitui um
  analisador linguístico completo.
- A amostra limitada não identifica necessariamente os trechos mais influentes
  de todo o artigo.
- O mDeBERTa usa os primeiros 256 tokens nesta versão. Remover uma frase pode
  trazer outros trechos para essa janela; o ramo léxico considera o texto completo.
- A explicação pode revelar vieses do modelo e efeitos do ajuste por texto curto.
  Ela não confirma fatos em fontes independentes.
- A fidelidade desta técnica e a latência precisam ser avaliadas em notícias reais
  de fontes e épocas variadas. Não há certificação causal nem avaliação humana
  de qualidade das explicações nesta entrega.
