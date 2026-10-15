# Modelos, dados e experimentos em outro ambiente

## Testar a aplicação

Os pesos treinados, o tokenizador e o manifesto do bundle ativo agora são
versionados por Git/Git LFS. Para testar localmente, basta clonar a branch
`main` (ou `modularization`), executar `git lfs pull` e seguir o [README](../README.md).
**Não é necessário um pacote de transferência para fazer inferência.**

| Conteúdo | Distribuição |
|---|---|
| API, interface, testes, scripts e `ml/` | Git |
| Classificadores sklearn, configurações, métricas e checksums do bundle | Git |
| Pesos mDeBERTa, SentencePiece e tokenizador serializado do bundle ativo | Git LFS |
| `banco_fake.csv` e ZIP histórico em `modelo/` | Git |
| `data/`, outros bundles, `mlflow.db` e `mlruns/` | Pacote de transferência opcional |
| `.venv*`, caches e `.env` | Recriar no destino |

## Exportar o estado de desenvolvimento

O pacote continua útil para transportar dados preparados, modelos experimentais
e o histórico MLflow. Ele não substitui a publicação dos objetos LFS no GitHub.
Finalize os commits e encerre treinamentos antes de exportar; a API pode continuar
executando inferências.

```bash
python3 scripts/transfer_artifacts.py export
```

O exportador gera, em `exports/` (ignorado pelo Git):

- `verificai-handoff-<instante>.tar.gz`: modelos, dados e tracking;
- `.tar.gz.sha256`: checksum do pacote;
- `.tar.gz.manifest.json`: inventário, modelo e commit de origem.

Ele usa backup consistente do SQLite e armazena conteúdos idênticos uma única vez.
Transfira os três arquivos por um meio adequado ao tamanho do pacote.

## Restaurar dados e tracking

O importador **recusa sobrescrever diretórios existentes**. Como o clone atual já
contém `models/mdeberta-ensemble/`, restaure o pacote em um diretório separado,
por exemplo `../verificai-restauracao`, que ainda não contenha esses artefatos:

```bash
python scripts/transfer_artifacts.py import \
  --archive /caminho/verificai-handoff-recebido.tar.gz \
  --root ../verificai-restauracao
```

Mantenha o `.sha256` ao lado do pacote. A importação verifica checksums e caminhos,
materializa cópias independentes dos pesos e migra os caminhos do MLflow para o
diretório de restauração. O registro `handoff-import.json` identifica a origem,
o commit e os checksums. Não faça extração manual se precisar restaurar o tracking.

Reserve espaço para os artefatos e uma área temporária equivalente; cerca de
7 GB livres atendiam ao pacote original, além do ambiente Python e do arquivo
compactado. O volume pode mudar com novos experimentos.

A API continua usando o bundle do clone, obtido pelo LFS. Para explorar os runs
restaurados, instale `requirements-lock.txt` e, a partir da raiz do clone, execute:

```bash
mlflow ui --backend-store-uri sqlite:///../verificai-restauracao/mlflow.db --host 127.0.0.1 --port 5002
```

Os runs principais são `658901685b1e42f88a0957f73a8e12e9` (fine-tuning) e
`8ab1f341b2604e19a894cfd209ca20e2` (seleção/calibração). A porta 5002 permite
consultar o MLflow enquanto a aplicação usa 5001.

## Retomar treinamento

O ambiente completo validado está em `requirements-lock.txt`: Python 3.14.6,
sklearn 1.8.0, torch 2.14.1 e transformers 4.57.6 no macOS arm64. Confira a
compatibilidade das wheels em outro sistema.

```bash
python -m pip install -r requirements-lock.txt
python -m pip check
python train_mdeberta.py --help
python train_ensemble.py --help
```

Informe ao CLI os caminhos dos dados restaurados ou prepare novamente `data/`,
seguindo o [guia de treinamento](mdeberta.md). Use um novo `--output-dir`, por
exemplo `models/mdeberta-experimento-02`; não sobrescreva o bundle publicado.
O cache do checkpoint inicial não faz parte do pacote e pode exigir novo download
para treinamento. A inferência do bundle publicado dispensa esse cache.

O pacote `ml/` mantém o motor do classificador importado. `ml/verification.py` é
legado, depende de `scraper.*` ausente e não participa dos endpoints atuais. Os
entrypoints suportados são `run.py`, `train_mdeberta.py` e `train_ensemble.py`.

## Publicar código e pesos

Siga [CONTRIBUTING.md](../CONTRIBUTING.md): revise o commit, confira os ponteiros
com `git lfs ls-files` e envie a branch com o hook LFS ativo. Dados, experimentos
e exportações locais continuam fora do Git. O envio ao remoto e o teste de um
clone remoto são etapas distintas da preparação local.
