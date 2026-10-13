# Continuar o desenvolvimento em outro ambiente

O código e os artefatos treinados são entregas complementares. A branch de
desenvolvimento é `modularization`. O Git contém a API, interface, testes, scripts,
documentação, `ml/`, o classificador importado e `banco_fake.csv`. O pacote de
transferência contém os pesos grandes e o histórico local dos experimentos.

## O que transferir

| Entrega | Conteúdo | Transferência |
|---|---|---|
| Repositório | Código, classificador importado, banco rotulado pelo responsável e versões das dependências | Git |
| `models/mdeberta-ensemble/` | Transformer ajustado, tokenizador, classificador, calibração, métricas e procedência | Pacote de artefatos |
| `models/ensemble/` | Ramo léxico treinado pelo CLI genérico | Pacote de artefatos, se presente |
| `data/` | Fake.Br original, revisão do corpus e dataset preparado | Pacote de artefatos, se presente |
| `mlflow.db` e `mlruns/` | Experimentos, parâmetros, histórico de métricas e artefatos | Pacote de artefatos, se presentes |
| `.venv*`, caches e `.env` | Instalações e configurações da máquina | Recriar no destino |

Não há DVC configurado. Os diretórios grandes continuam ignorados pelo Git.
O pacote inclui a revisão Git, a versão do modelo e SHA-256 de cada arquivo.
Ele preserva os IDs dos runs do MLflow. A importação ajusta as localizações dos
artefatos no SQLite para a nova raiz, sem alterar parâmetros ou métricas.

## Preparar a transferência na origem

Finalize os commits e encerre treinamentos antes da exportação. A API pode
continuar executando inferências. O exportador usa backup consistente do SQLite
e compacta apenas uma cópia de conteúdos idênticos, incluindo os pesos duplicados
no MLflow.

```bash
python3 scripts/transfer_artifacts.py export
```

O comando gera três arquivos em `exports/`, com o mesmo prefixo:

- `verificai-handoff-<instante>.tar.gz`: pacote de dados, modelos e tracking;
- `.tar.gz.sha256`: checksum do pacote;
- `.tar.gz.manifest.json`: inventário, versão do modelo e commit correspondente.

Leve os três arquivos ao outro ambiente por um meio adequado ao tamanho do pacote.
`exports/` é local e não entra no `git push`. O exportador não sobrescreve pacotes.
Para enviar os commits desta branch ao remoto configurado:

```bash
git push -u origin modularization
```

## Restaurar no destino

O ambiente validado utiliza **Python 3.14.6**, com `scikit-learn==1.8.0`,
`torch==2.14.1`, `transformers==4.57.6` e `mlflow==3.17.1`.
`requirements-lock.txt` registra todas as versões instaladas no macOS arm64.
O arquivo `.python-version` pode ser usado por um gerenciador de Python.
Em outro sistema operacional, a disponibilidade das mesmas wheels precisa ser
confirmada; a instalação e a execução devem passar pelas verificações abaixo.

```bash
git clone --branch modularization git@github.com:alvezclari/verificAI.git
cd verificAI

# Use o commit indicado no .tar.gz.manifest.json para repetir exatamente a versão.
# git checkout <git_commit>

python3.14 -m venv .venv-classificador
source .venv-classificador/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check

# Substitua pelo caminho do pacote recebido; mantenha o .sha256 ao lado dele.
python scripts/transfer_artifacts.py import \
  --archive /caminho/verificai-handoff-recebido.tar.gz
```

Use um clone sem `data/`, `mlruns/`, `mlflow.db` ou os bundles de destino. A
importação verifica checksums e caminhos antes de publicar os arquivos e recusa
sobrescrever artefatos existentes. Use o script para restaurar: ele materializa
cópias independentes dos pesos e migra os caminhos do MLflow. A extração manual
com `tar` não realiza essas etapas. Reserve espaço para os arquivos restaurados
e uma área temporária equivalente; nesta versão, cerca de 7 GB livres atendem à
restauração, além do ambiente Python e do pacote compactado.

`handoff-import.json`, ignorado pelo Git, registra origem, destino, commit,
checksums e instante de restauração. O SQLite tem seu checksum alterado pela
migração de caminhos; o checksum do pacote original permanece no registro.

## Validar e executar a API

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
node --check frontend/app.js

cp .env.example .env
set -a
source .env
set +a
python run.py
```

O Node é necessário apenas para essa checagem de sintaxe; o frontend é servido
pelo Flask. A aplicação não carrega `.env` automaticamente. O exemplo seleciona
`MODEL_BACKEND=ensemble` explicitamente, permitindo detectar uma restauração
incompleta. `TRANSFORMER_DEVICE=auto` escolhe CUDA, MPS ou CPU. CPU permite executar
sem GPU; o tempo de resposta depende do hardware.

Em outro terminal:

```bash
curl http://127.0.0.1:5000/api/health

curl -X POST http://127.0.0.1:5000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://portal.com.br/caminho-de-uma-noticia-acessivel"}'
```

No health, confira `status=online`, backend `ensemble`, Transformer carregado e versão
`mdeberta-ensemble-20261013T185545`. Na análise, confira
`ml_details.component_probabilities`, os dois pesos, `prob_fake`, `faixa` e
`model_evaluation.splits.test`. Use uma URL real no lugar do exemplo.
Consulte a interface em `/` e o contrato interativo em `/docs`.

A inferência dos modelos é local. A análise por URL requer acesso à página;
bloqueios, paywalls e alterações da notícia podem mudar a extração. A persistência
do conteúdo extraído e o replay ainda estão pendentes.

Para consultar os experimentos restaurados, execute na raiz:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Os runs principais permanecem `658901685b1e42f88a0957f73a8e12e9` (fine-tuning) e
`8ab1f341b2604e19a894cfd209ca20e2` (seleção/calibração).

## Retomar o treinamento

O checkpoint inicial do Hub pode ser baixado novamente na revisão fixada; seu
cache local não faz parte do pacote. Os dados preparados e o bundle já treinado
permitem continuar a inferência sem esse download.

```bash
python train_mdeberta.py --help

python train_mdeberta.py \
  --base-model MoritzLaurer/mDeBERTa-v3-base-mnli-xnli \
  --revision 8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c \
  --source-model MoritzLaurer/mDeBERTa-v3-base-mnli-xnli \
  --source-revision 8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c \
  --replace-classification-head \
  --output-dir models/mdeberta-experimento-02

python train_mdeberta.py --calibrate-existing \
  --output-dir models/mdeberta-experimento-02
```

Use `--cpu` para forçar CPU no treinamento. Instalações CUDA precisam de PyTorch
compatível com o driver disponível; registre qualquer mudança do ambiente no
novo experimento. Um bundle completo não é sobrescrito pelo treino.
Os diretórios de novos experimentos também ficam ignorados pelo Git.

O pacote `ml/` preserva o código recebido com o classificador. Seus módulos de
inferência e pré-processamento são usados pela API. `ml/verification.py` é um
módulo legado que depende de `scraper.*`, ausente neste repositório; ele não
participa dos endpoints atuais. Os entrypoints suportados para esta arquitetura
são `run.py`, `train_mdeberta.py` e `train_ensemble.py`.

As próximas entregas são persistência/replay das extrações (M08) e validação
externa por fonte/época (M13), conforme o [acompanhamento](evolucao-projeto.md).
O ensemble continua experimental, com `production_ready=false`.
