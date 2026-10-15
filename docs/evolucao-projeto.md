# Evolução do VerificAI: marcos e estado atual

**Atualização:** 15/10/2026

**Escopo:** baseline, novo classificador calibrado, análise por URL, ensemble e MLOps.

**Referência técnica:** [diagnóstico e guia do ensemble](ensemble.md).

## Preparação para distribuição pelo GitHub — 15/10/2026

O bundle ativo foi preparado para versionamento com Git LFS: pesos, SentencePiece
e tokenizador serializado usam LFS; configurações, métricas e checksums usam Git.
O README descreve clone, inferência e limitações; `scripts/check_model.py` detecta
ponteiros não baixados e verifica integridade. Protótipos anteriores foram reunidos
em `archive/`. A publicação é feita em `main` e `modularization`, conforme
[CONTRIBUTING.md](../CONTRIBUTING.md). O roteiro inclui uma instalação limpa
e `scripts/smoke_test.py` para verificar interface, API, extração e os dois
modelos reais em CPU, sem downloads durante a inferência.

## Situação atual

A API agora usa o ensemble supervisionado em `models/mdeberta-ensemble/`,
combinando o classificador calibrado importado com mDeBERTa-v3. O fine-tuning,
a calibração e a seleção dos pesos foram executados e registrados no MLflow.
O health confirmou os dois componentes carregados e o Transformer em GPU MPS.

A interface mostra as probabilidades separadas, os pesos, o resultado combinado,
a faixa e as métricas de teste. A API também informa a cobertura do texto no ramo
semântico: até 256 tokens nesta versão. O classificador isolado continua disponível
com `MODEL_BACKEND=calibrated`, e o baseline com `MODEL_BACKEND=legacy`.

O teste reservado de 1.659 registros produziu F1 macro de 0,8773 para o Transformer
e 0,8814 para o ensemble. O experimento utilizou 8.295 textos do Fake.Br e do
`banco_fake.csv`, cujos rótulos falsos foram confirmados pelo responsável.
Veja o [relatório do mDeBERTa](mdeberta.md) e o [guia da API local](api-local.md).

As entregas estão organizadas em commits locais por área: API/interface,
treinamento/MLOps, portabilidade e documentação. O envio ao remoto e o deploy
são etapas posteriores. O [guia de continuação](continuacao-em-outro-ambiente.md)
descreve como transferir os artefatos e reconstruir o ambiente.

## Estados utilizados

| Estado | Significado |
|---|---|
| Concluído | Entrega implementada ou investigação encerrada, com evidência indicada. |
| Validado isoladamente | Componente executado com sucesso, ainda sem conexão ao fluxo completo da aplicação. |
| Parcial | Parte da entrega está pronta, mas faltam etapas para cumprir o objetivo completo. |
| Pendente | Trabalho identificado e ainda não executado neste repositório. |
| Opcional | Evolução possível, sem infraestrutura existente que a torne obrigatória. |

## Marcos do desenvolvimento

| Marco | Entrega | Estado | Evidência ou pendência principal |
|---|---|---|---|
| M01 | Investigar dados, processamento e baseline | Concluído | Scripts, notebooks no ZIP, artefatos e inferência mapeados; resumo emitido no terminal. |
| M02 | Definir a arquitetura de ensemble | Concluído | Dois ramos em paralelo, classes alinhadas e soft voting configurável. |
| M03 | Implementar integração HuggingFace e ensemble | Concluído | Módulos em `backend/ml/`; fine-tuning e inferência validados com modelo pequeno local. |
| M04 | Adaptar treinamento e rastreamento | Concluído | CLI com etapas separadas/integradas, splits compartilhados e MLflow. |
| M05 | Executar o novo treinamento léxico no corpus disponível | Concluído | 11.902 registros; métricas e artefatos registrados no MLflow. |
| M06 | Inspecionar o `classificador.joblib` adicionado | Validado isoladamente | Motor original carregou o artefato, validou o pré-processamento e repetiu previsões idênticas. |
| M07 | Integrar o classificador calibrado à API por URL | Concluído | Adaptador original, seleção explícita, ambiente compatível, faixas e explicações na API/frontend. |
| M08 | Registrar extrações de URL para reprodução | Parcial | Resposta inclui hashes, instantes, URL final e versões; persistência do conteúdo e replay continuam pendentes. |
| M09 | Ajustar mDeBERTa em textos originais rotulados | Concluído | Fine-tuning das últimas duas camadas e cabeça binária; Fake.Br + CSV confirmado, com treino/validação/calibração/teste agrupados. |
| M10 | Avaliar o classificador calibrado junto ao Transformer | Concluído | Comparação experimental no teste reservado e seleção/calibração separadas; independência cruzada completa do artefato histórico ainda não certificada. |
| M11 | Ativar a versão escolhida na aplicação | Concluído | Ensemble `mdeberta-ensemble-20261013T185545` ativo na API local; POST real e health confirmados. |
| M12 | Versionar os bundles com DVC | Opcional | Não existe configuração DVC ou remote neste repositório. |
| M13 | Validar generalização e preparar produção | Pendente | Avaliação externa por fonte/época, conteúdo curto e independência cruzada dos modelos; artefato continua experimental. |
| M14 | Preparar continuação em outro ambiente | Concluído | Versões fixadas, configuração de exemplo, exportação/importação com SHA-256, migração dos caminhos MLflow e guia de restauração. |

M07 a M11 estão concluídos para o protótipo local. M08 e M13 delimitam as
próximas entregas de reprodução e validação externa. M14 permite transferir o
estado treinado sem versionar os pesos grandes no Git.

## Relação com o plano original

| Fase | Estado | O que já foi entregue | O que ainda falta |
|---|---|---|---|
| 1 — Investigação e diagnóstico | Concluído | Localização dos dados/notebooks, parâmetros dos modelos, fluxo de treino/inferência e divergências de rótulos. | Atualizar o diagnóstico quando novos dados ou artefatos forem adicionados. |
| 2 — Planejamento do ensemble | Concluído | TF-IDF + LogReg e Transformer em paralelo; soft voting; rótulos canônicos `0=fake, 1=confiável`. | Faixas implementadas; futuros limiares exigem nova avaliação. |
| 3 — Execução e refatoração | Concluído | Carregamento/tokenização/inferência semântica, combinação, CLI de fine-tuning e suporte ao manifesto na API. | Integração local concluída; generalização externa segue em M13. |
| 4 — Rastreamento e MLOps | Parcial | MLflow para novos treinos, parâmetros, métricas, hashes, splits e artefatos. | Procedência importada e bundle registrados no MLflow; faltam snapshots/replay e consolidação para produção. DVC é opcional. |

## Inventário dos modelos e componentes

| Componente | Localização | Estado atual |
|---|---|---|
| Baseline LinearSVC | `models/best_linearsvc_model.joblib` | Disponível no modo `MODEL_BACKEND=legacy`. |
| Artefato legado adicional | `models/model.pkl` | Preservado; tem menor prioridade que o LinearSVC. |
| LogReg treinada pelo novo CLI | `models/ensemble/lexical.joblib` | Treinada e registrada no MLflow; não está ativa na API. |
| Classificador calibrado adicionado | `models/classificador.joblib` | Ramo léxico do ensemble; disponível isoladamente com `MODEL_BACKEND=calibrated`. |
| Regras do classificador calibrado | `ml/artifact.py`, `ml/calibration.py`, `ml/preprocessing.py`, `ml/inference.py` | Presentes; o motor original foi executado com esse código. |
| Ensemble e Transformer | `backend/ml/deep_ensemble.py`, `backend/ml/transformer.py` | Ajustados, calibrados e ativos localmente; pesos em `models/mdeberta-ensemble/transformer/`. |
| CLI de treinamento | `train_ensemble.py`, `backend/ml/training.py` | Etapas `lexical`, `transformer`, `all` e `ensemble` disponíveis. |
| Manifesto do ensemble ativo | `models/mdeberta-ensemble/manifest.json` | Presente, schema v2, com pesos, calibração, métricas e versões. |
| CLI do mDeBERTa com artefato importado | `train_mdeberta.py` | Fine-tuning e etapa `--calibrate-existing` executados. |

`models/ensemble/lexical.joblib` e `models/classificador.joblib` são artefatos
distintos. O novo CLI `train_mdeberta.py` importa o segundo sem retreinar seus pesos.

## Características do classificador adicionado

O arquivo contém `ArtefatoClassificador`, com uma pipeline de TF-IDF de palavras
e caracteres, Regressão Logística, calibrador, coeficiente de contração de
probabilidades e manifesto de treinamento.

| Item | Configuração identificada |
|---|---|
| Classes | `0=falsa`, `1=verdadeira` |
| Entrada | Título e corpo, tratados por `ml.preprocessing.preparar_texto` |
| TF-IDF de palavras | N-gramas de 1 a 2; até 20.000 atributos |
| TF-IDF de caracteres | N-gramas de 3 a 5; até 30.000 atributos |
| Regressão Logística | `C=2`, `max_iter=2000`, seed 42 |
| Calibração do artefato | Sigmoide ajustada |
| Contração para 0,5 | `r ≈ 0,935074` |
| Texto curto | Confiança reduzida proporcionalmente abaixo de 33 palavras preparadas |
| Versão do modelo | `texto-5c379316b7da-20261013T123124` |
| Ambiente registrado | Python 3.14.6; sklearn 1.8.0; NumPy 2.4.2; SciPy 1.17.1; joblib 1.5.3; pandas 3.0.1; tldextract 5.3.1 |

O motor também valida o hash do pré-processamento e a versão do scikit-learn.
O `.venv` dos experimentos usa sklearn 1.9.1. O ambiente separado
`.venv-classificador` foi preparado com sklearn 1.8.0 e as versões listadas em
`requirements-classificador.txt`, permitindo servir o artefato pela API.

As faixas originais usam a probabilidade final de fake news:

| Condição | Faixa |
|---|---|
| `prob_fake <= 0,35` | `verdadeira` |
| `0,35 < prob_fake <= 0,65` | `inconclusiva` |
| `prob_fake > 0,65` | `falsa` |

A integração na aplicação mantém a
calibração, a redução de confiança para textos curtos e essas faixas. A pontuação
correspondente de confiabilidade é `100 × (1 − prob_fake)`. A mistura de
70% ML com 30% heurísticas fica restrita ao modo legado.

## Evidências históricas anteriores à integração final

| Verificação | Resultado | Limite da evidência |
|---|---|---|
| Primeira execução registrada da suíte do ensemble | 23 testes aprovados em modo offline, incluindo os oito testes originais da API | Naquele marco, a integração do classificador importado ainda estava pendente. |
| Fine-tuning semântico local | BERT pequeno criado localmente, treinado, salvo e recarregado; ensemble e MLflow verificados | Valida o fluxo técnico; não mede qualidade do mDeBERTa em português. |
| Treino léxico no CSV do ZIP | 7.140/2.381/2.381 registros de treino/validação/teste; F1 macro de teste `0,959679`; ROC AUC `0,991046` | Métricas do ramo léxico treinado pelo novo CLI; não são do classificador importado nem do ensemble final. |
| Run MLflow desse treino léxico | `62980323bb964a3ebad59112e9616e51` | Identifica o experimento local; não representa promoção para produção. |
| Carregamento do classificador importado | `MotorClassificador.ready=True`, sem erro, usando o pacote `ml/` deste repositório e o ambiente original | Nesse primeiro teste, apenas o motor isolado foi exercitado. |
| Repetição de uma entrada de exemplo | Duas respostas idênticas; `prob_fake ≈ 0,03588`, faixa `verdadeira` | Teste de execução e determinismo; não prova acerto factual ou desempenho geral. |
| Consulta local ao health da API | HTTP 200, modelo carregado e `ensemble_loaded=false` | Naquele marco, a configuração ainda selecionava o LinearSVC. |

O corpus exportado disponível contém corpo já transformado e pistas de classe,
como `#boato` nos títulos. As métricas aleatórias precisam ser interpretadas com
essa limitação. O experimento posterior do mDeBERTa utiliza o novo corpus descrito
no relatório, com métricas próprias.

## Evidências da integração da API

Validação após a integração: **37 testes aprovados e 2 pulados** no ambiente
do classificador; **23 aprovados e 16 pulados** no ambiente do ensemble. Os skips
separam as dependências e versões próprias de cada artefato.

- `MotorClassificador` é utilizado pelo adaptador `backend/ml/calibrated.py`.
- O score da API coincide com o motor original e não inclui as heurísticas.
- As faixas verdadeira/inconclusiva/falsa foram verificadas nos limites 0,35 e 0,65.
- O health identifica backend, versão, hash e erro; falhas do modelo retornam 503.
- A interface Flask apresenta probabilidades, contribuições e métricas do treinamento.
- `/docs` permite testar URLs e `/api/openapi.json` fornece o contrato HTTP.
- O teste com notícia pública extraiu 403 palavras e retornou HTTP 200; os detalhes
  constam no [guia da API local](api-local.md).

Os resultados anteriores da tabela de evidências permanecem como histórico;
a integração do classificador, antes pendente, está concluída localmente.

## Próximas entregas e critérios de conclusão

### 1. Conectar o classificador calibrado ao fluxo de URL — concluído

- Preparar dependências compatíveis com o manifesto do artefato.
- Criar um adaptador para `MotorClassificador.prever(titulo, corpo)`.
- Permitir seleção explícita de modelo, preservando o baseline como alternativa.
- Retornar `prob_fake`, score correspondente, faixa, explicações e versão do modelo.
- Identificar o modelo ativo e erros de carregamento no endpoint de health.
- Definir a apresentação de `inconclusiva` na API e no frontend.

**Critério de conclusão:** para o mesmo título/corpo, o motor isolado e a API
retornam a mesma probabilidade final e faixa; falhas do modelo ficam explícitas.

### 2. Tornar a análise de URL reproduzível — parcial

- Registrar URL recebida e URL final, instante da coleta, título e corpo extraídos.
- Registrar hash do conteúdo, versão do modelo e versões do scraper/pré-processamento.
- Criar casos de teste com conteúdo salvo, sem depender de páginas externas.
- Separar testes de extração, testes de classificação e testes de rede.

**Critério de conclusão:** uma análise pode ser repetida a partir do conteúdo salvo
e da versão do modelo, mesmo que a página original tenha mudado. A API já fornece hashes e metadados de extração, mas não
persiste esses snapshots.

### 3. Evoluir para o ensemble com o novo artefato — concluído localmente

- Revisar o corpus e obter textos originais adequados ao treinamento semântico.
- Preservar a separação entre treino, calibração, validação e avaliação final.
- Adaptar a saída calibrada do classificador para `[P(fake), P(real)]`.
- Ajustar o Transformer e registrar o experimento no MLflow.
- Compatibilizar a procedência dos artefatos importados com o manifesto do ensemble.
- Comparar baseline, classificador calibrado, Transformer e ensemble em dados
  independentes dos usados para ajuste e seleção dos modelos.
- Escolher pesos usando validação e medir latência, memória e desempenho final.

**Critério de conclusão:** bundle completo, rastreável e recarregável, com avaliação
comparativa registrada e política definida para resultados inconclusivos.

O fluxo de importação foi implementado com manifesto v2 e identificação por hash
do classificador, mantendo suas regras originais como componente do ensemble.

### 4. Consolidar MLOps e ativação

- Registrar no MLflow a importação do classificador, seu hash e seu manifesto,
  preservando a origem das métricas de treinamento existentes.
- Versionar código, dependências e bundle completo; adotar DVC se necessário.
- Validar ativação, recarregamento e retorno ao baseline.
- Atualizar este documento com a evidência de cada marco concluído.

**Critério de conclusão:** a versão ativa pode ser identificada, reproduzida e
substituída por uma versão anterior sem ambiguidade sobre seus artefatos.

## Evidências da integração final com mDeBERTa

- 46 testes aprovados no ambiente unificado do classificador e Transformer.
- Fine-tuning registrado no run `658901685b1e42f88a0957f73a8e12e9`.
- Seleção/calibração registrada no run `8ab1f341b2604e19a894cfd209ca20e2`.
- Pesos selecionados na validação: 90% saída léxica e 10% saída semântica, com
  temperatura/intercepto ajustados na reserva de calibração.
- Teste: F1 macro 0,8814, ROC AUC 0,9606, log loss 0,2715 e Brier 0,0827 para o ensemble.
- 225 registros do banco reservados para teste: recall falso binário de 84,89%;
  80,89% na faixa falsa e 8% inconclusivos.
- POST real à API: HTTP 200, ambos os componentes presentes, score combinado
  89,3191 no exemplo do G1. O score do classificador isolado nesse conteúdo era 96,7505.
- Health: backend `ensemble`, Transformer carregado em `mps`, tarefa supervisionada.

Esses resultados pertencem ao novo protocolo e não devem ser comparados diretamente
com o F1 de 95,97% do treino léxico anterior. O modelo continua experimental.

## Como manter este acompanhamento

Ao concluir uma entrega, atualizar seu estado, data e evidência: teste executado,
run MLflow, artefato ou commit correspondente. Registrar métricas junto ao modelo,
dataset e protocolo que as produziram. Uma implementação validada com dados de
exemplo não deve ser marcada como avaliação concluída do modelo real.
