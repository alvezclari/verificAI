# API local com o classificador calibrado

O fluxo local atual é URL → extração do título/corpo → classificador calibrado
+ mDeBERTa-v3 → probabilidade combinada, faixa, contribuições e métricas.
O [guia do mDeBERTa](mdeberta.md) descreve dados, fine-tuning e avaliação.

## Executar

Na raiz do repositório, com Python 3.14:

```bash
python3.14 -m venv .venv-classificador
source .venv-classificador/bin/activate
python -m pip install -r requirements-lock.txt
python run.py
```

Abra [a aplicação](http://127.0.0.1:5000/) para inserir a URL de uma notícia ou
[a documentação interativa](http://127.0.0.1:5000/docs) para consultar a resposta JSON.
O servidor local usa `127.0.0.1:5000`, sem debug por padrão. `PORT`, `APP_HOST` e
`FLASK_DEBUG` permitem ajustar a execução.

Ao clonar em outra máquina, restaure também o pacote de pesos/dados, conforme o
[guia de continuação](continuacao-em-outro-ambiente.md), para executar com mDeBERTa.

O ambiente separado preserva o `.venv` anterior e agora inclui as dependências do Transformer. As versões
de sklearn, NumPy, SciPy, joblib, pandas e tldextract são as registradas no artefato.
O motor valida a versão do sklearn e o hash do pré-processamento. Não altere
`ml/preprocessing.py` sem gerar um artefato compatível.

## Seleção do modelo

| Variável | Padrão | Uso |
|---|---|---|
| `MODEL_BACKEND` | `ensemble` com bundle completo; caso contrário, `calibrated` | `calibrated`, `legacy` ou `ensemble` |
| `CLASSIFIER_PATH` | `models/classificador.joblib` | Caminho do artefato calibrado |
| `ENSEMBLE_MANIFEST` | `models/mdeberta-ensemble/manifest.json` quando presente | Bundle usado com `MODEL_BACKEND=ensemble` |
| `TRANSFORMER_DEVICE` | `auto` (CUDA/MPS/CPU) | Dispositivo usado no modo ensemble |

O modo `legacy` preserva a seleção antiga do LinearSVC. O modo `ensemble` exige
bundle completo e dependências semânticas. Na inicialização sem `MODEL_BACKEND` explícito, a presença do bundle completo
`models/mdeberta-ensemble` seleciona o ensemble. Um processo já iniciado precisa
ser reiniciado para essa seleção automática. Alterações de variáveis exigem reinício;
`POST /api/reload-model` recarrega os arquivos dentro da configuração já iniciada.

## Endpoints

- `GET /api`: atalhos para documentação e estado do serviço.
- `GET /api/health`: backend, caminho, versão, hash e erro do modelo selecionado.
- `GET /api/openapi.json`: contrato OpenAPI 3.0.3.
- `POST /api/analyze`: recebe um objeto JSON com `url`.
- `POST /api/reload-model`: recarrega o modelo; retorna 503 quando falha.

```bash
curl http://127.0.0.1:5000/api/health

curl -X POST http://127.0.0.1:5000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://exemplo.com.br/noticia"}'
```

Troque a URL de exemplo por uma notícia pública acessível. A API não analisa URLs
de localhost. Páginas com bloqueio, paywall ou sem texto suficiente podem falhar.

## Interpretação dos resultados

| Campo | Significado |
|---|---|
| `prob_fake` | Probabilidade final da classe falsa, após calibração e ajustes originais |
| `confidence_score` | `100 × (1 − prob_fake)`, sem mistura com heurísticas |
| `faixa` | `verdadeira`, `inconclusiva` ou `falsa` |
| `verdict` | Alias da aplicação: `confiavel`, `inconclusiva` ou `desconfiavel` |
| `thresholds` | Limiares efetivamente carregados do manifesto |
| `model_version` | Versão do classificador que produziu a resposta |
| `ml_details.explicacao` | Termos com contribuições favoráveis/desfavoráveis à classe verdadeira |
| `ml_details.fator_texto_curto` | Redução de confiança para textos preparados com menos de 33 palavras |
| `model_evaluation` | Métricas por componente/partição no ensemble; metadados históricos no modo calibrado isolado |
| `criteria` | Cinco sinais heurísticos auxiliares |
| `criteria_affect_score` | `false` no classificador calibrado e no ensemble |
| `provenance` | Instantes da análise/extração, hash da entrada/modelo, URL final e versões do extrator |

As faixas do artefato atual são: P(falsa) ≤ 0,35 → verdadeira; 0,35 < P(falsa) ≤ 0,65
→ inconclusiva; P(falsa) > 0,65 → falsa. `threshold` fica nulo nesse modo, pois a
decisão usa dois limiares. A interface aplica essas faixas antes de arredondar o score.

No modo calibrado isolado, as métricas vêm do manifesto do classificador importado. O F1 macro de CV
do treino é aproximadamente 96,13%; isso não é o F1 de teste do outro ramo léxico
treinado pelo CLI, nem uma medida de acerto da URL enviada. O artefato usa protocolo
experimental e declara `production_ready=false`. Os corpos do corpus já foram
transformados e as fontes podem aparecer em mais de uma partição.

A classificação considera padrões textuais. As contribuições e os cinco sinais
auxiliares não confirmam fatos em fontes externas. A interface padrão servida pelo
Flask apresenta esses resultados em `frontend/`.

## Erros

| HTTP | Condição |
|---|---|
| 400 | URL inválida, JSON inválido ou entrada diferente de objeto JSON |
| 422 | Falha ao acessar ou extrair o conteúdo da notícia |
| 503 | Dependências incompatíveis, artefato ausente/inválido ou falha de inferência |
| 500 | Erro interno inesperado |

No modo calibrado não é produzido um score substituto quando o modelo falha.
O health continua consultável e informa `status=degraded` e o diagnóstico.

## Testes e evidências

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-classificador/bin/python -m pytest -q
node --check frontend/app.js
```

O ambiente completo de `requirements-lock.txt` reúne o classificador e as
dependências do Transformer/MLflow. A suíte também verifica a transferência dos
artefatos, a integridade dos arquivos e a migração dos caminhos do tracking.

A suíte do classificador usa HTML controlado com extração real e compara a API
com o motor original. Testa limites das faixas, ajuste de texto curto, repetição de
resultados, incompatibilidade de versão e erros de extração/inferência. Testes
que exigem sklearn 1.8.0 ou dependências semânticas são pulados se o ambiente
não for compatível; use o ambiente completo para executar toda a suíte.

Em 13/10/2026, o fluxo também foi executado com uma notícia pública do G1:
403 palavras extraídas, HTTP 200, `prob_fake=0,03249500327099408`, score
`96,75049967290059` e faixa `verdadeira`, com a versão
`texto-5c379316b7da-20261013T123124`. Isso valida a execução com rede; não confirma a
verdade factual da matéria nem garante repetição se a página mudar.

O hash da entrada facilita a comparação entre execuções, mas ainda não há
persistência automática do corpo extraído ou replay de snapshots. Essa entrega
continua no [acompanhamento da evolução](evolucao-projeto.md).


## Atualização: participação do mDeBERTa

O bundle `mdeberta-ensemble-20261013T185545` foi ativado localmente. O health
identifica o Transformer e a resposta inclui `ml_details.component_probabilities`,
`weights`, `temperature`, `bias` e `transformer`. A interface apresenta os dois
componentes e avisa quando o texto é truncado no ramo semântico.

No teste de 1.659 registros, o F1 macro
foi 0,8773 para mDeBERTa e 0,8814 para o ensemble. As métricas históricas do
classificador isolado, citadas acima, pertencem a outro protocolo. Consulte o
[relatório do ensemble](mdeberta.md) para a comparação e as limitações.
