# VerificAI

> **Antes de compartilhar, observe os sinais.**

A **VerificAI** é um protótipo de aplicação voltado à **avaliação da confiabilidade de informações e notícias**. A proposta é auxiliar o usuário na análise crítica de uma informação antes de seu compartilhamento, organizando diferentes sinais de evidência em um **indicador geral de confiabilidade**.

O sistema recebe o link de uma notícia e apresenta uma análise estruturada com base em cinco critérios, permitindo que o usuário compreenda **por que uma informação recebeu determinada avaliação**, em vez de depender apenas de uma classificação automática.

A versão atual combina classificação textual léxica e semântica, com cinco
sinais heurísticos auxiliares. A verificação de alegações em fontes independentes
faz parte da evolução prevista.

## Objetivo

A aplicação tem como objetivo apoiar a verificação crítica de informações na internet, oferecendo uma visão consolidada sobre a qualidade das evidências relacionadas a uma notícia.

- analisar evidências relacionadas a uma informação;
- avaliar a qualidade das fontes utilizadas;
- verificar a existência de corroboração por fontes independentes;
- considerar o contexto apresentado pela informação;
- considerar a atualidade e a relevância temporal das fontes;
- apresentar um indicador geral de confiabilidade;
- tornar os critérios utilizados na avaliação transparentes para o usuário.


## Proposta

A aplicação parte de um princípio simples:

> **Uma informação não deve ser considerada confiável apenas por parecer convincente. É necessário examinar as evidências que a sustentam.**

Por isso, em vez de apresentar somente um resultado final, a VerificAI detalha os fatores que contribuíram para a avaliação.

O usuário insere o endereço de uma notícia no campo de entrada e seleciona **"Verificar"**. A aplicação então organiza os sinais encontrados e apresenta uma avaliação explicável.

## Princípios da aplicação

A VerificAI foi concebida seguindo alguns princípios:

### Transparência

O usuário consegue visualizar os critérios utilizados para chegar ao resultado.

### Explicabilidade

A aplicação apresenta os fatores que sustentam e enfraquecem a avaliação.

### Criticidade

O sistema não pretende substituir o julgamento humano.

### Corroboração

A confirmação por fontes independentes é tratada como um elemento importante da análise.

### Atualidade

A relevância temporal das informações é considerada no processo.

### Responsabilidade

O indicador é apresentado como uma **estimativa**, evitando transmitir uma falsa certeza sobre a veracidade de uma informação.

## Execução

```bash
python3.14 -m venv .venv-classificador
source .venv-classificador/bin/activate
python -m pip install -r requirements-lock.txt
python run.py
```

Acesse `http://localhost:5000`. O Flask serve o frontend em `frontend/`.
Quando o bundle treinado está presente, o padrão é o ensemble de
`classificador.joblib` com mDeBERTa-v3, com probabilidades por componente,
calibração e faixa inconclusiva. Sem esse bundle, o padrão é o classificador
calibrado isolado. Acesse `/docs` para testar a API e consultar o
JSON. Veja o [guia da API local](docs/api-local.md).

Para levar os pesos treinados, os dados e o histórico MLflow a outra máquina,
siga o [guia de continuação em outro ambiente](docs/continuacao-em-outro-ambiente.md).
O clone contém o classificador importado; o mDeBERTa treinado é restaurado pelo
pacote de artefatos, fora do Git.

## Modelos e ensemble

A API mantém os artefatos LinearSVC existentes e agora aceita um ensemble de
**TF-IDF + Regressão Logística e Transformer mDeBERTa-v3**, com inferência paralela
e soft voting de probabilidades. Um manifesto local seleciona o ensemble após
fine-tuning com `MODEL_BACKEND=ensemble`. `MODEL_BACKEND=legacy` seleciona o
baseline; `calibrated` usa o classificador importado isoladamente.

O treinamento suporta etapas separadas ou integradas, splits compartilhados e
registro obrigatório de hiperparâmetros, métricas, dados e artefatos no MLflow.

```bash
pip install -r requirements-ml.txt
python train_ensemble.py --help
```

Veja o [diagnóstico e guia completo do ensemble](docs/ensemble.md), incluindo
convenções de rótulos, limitações do dataset, treinamento, MLflow e ativação.

Acompanhe os [marcos, estados atuais e próximas entregas](docs/evolucao-projeto.md)
da evolução do projeto.

A integração supervisionada com o corpus `banco_fake.csv`, o Fake.Br e o
classificador importado está descrita no [guia do mDeBERTa na API](docs/mdeberta.md).

## API

- `GET /api/health`: estado dos modelos, ensemble e erros de carregamento.
- `POST /api/analyze`: recebe `{"url":"https://..."}` e retorna análise, score,
  veredito, critérios e detalhes dos modelos.
- `POST /api/reload-model`: recarrega os artefatos locais.

No classificador calibrado, o score é `100 × (1 − prob_fake)`, com as faixas
originais verdadeira/inconclusiva/falsa. No ensemble supervisionado v2, o score é
a probabilidade combinada e calibrada de notícia confiável × 100, com as faixas
definidas por P(fake) de 0,35 e 0,65; os cinco critérios continuam informativos.
O ensemble genérico v1 usa limiar 50. No caminho legado,
o score mantém a combinação de 70% ML e 30% critérios. A sigmoide da margem do
LinearSVC legado é uma aproximação sem calibração aprendida.

## Testes

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

Com `requirements-ml.txt`, a suíte também testa fine-tuning local, inferência do
ensemble e rastreamento real no MLflow, sem baixar um modelo durante os testes.
