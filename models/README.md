# Modelos e procedência

O bundle publicado é **`mdeberta-ensemble-20261013T185545`**, em
`models/mdeberta-ensemble/`. Ele combina o classificador calibrado importado com
mDeBERTa ajustado para a tarefa binária `0=fake, 1=confiavel`.

## Baixar e conferir

Na raiz do clone, com Git LFS instalado:

```bash
git lfs install --local
git lfs pull
python scripts/check_model.py --verify-hashes
```

Não é necessário restaurar MLflow, baixar datasets ou retreinar para executar a
API. Consulte o [README principal](../README.md).

## Arquivos do bundle

| Arquivo | Finalidade | Distribuição |
|---|---|---|
| `manifest.json` | Classes, pesos, calibração, faixas e métricas | Git |
| `checksums.json` | Tamanhos e SHA-256 dos 11 arquivos necessários | Git |
| `data_provenance.json` | Partições e procedência do experimento | Git |
| `classificador.joblib` | Classificador calibrado alinhado ao manifesto | Git |
| `transformer/model.safetensors` | Pesos supervisionados, aproximadamente 1,04 GiB | Git LFS |
| `transformer/spm.model` | Vocabulário SentencePiece | Git LFS |
| `transformer/tokenizer.json` | Tokenizador serializado | Git LFS |
| `transformer/config.json` | Arquitetura e mapeamento de classes | Git |
| Demais JSONs em `transformer/` | Tokens especiais e metadados do treino/tokenizador | Git |

`training_args.bin` não é necessário para inferência e continua local. Os outros
experimentos em `models/ensemble/` e `models/mdeberta-*/` continuam ignorados.
Ao substituir o bundle, atualize o inventário junto aos pesos e metadados.

## Origem do modelo NLP

- Checkpoint inicial: [MoritzLaurer/mDeBERTa-v3-base-mnli-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli).
- Revisão: `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c`.
- Ajuste supervisionado para notícias, cabeça binária e 10 camadas congeladas.
- Corpus: 8.295 registros do [Fake.Br](https://github.com/roneysco/Fake.br-Corpus)
  e do `banco_fake.csv`; os rótulos falsos deste último foram confirmados pelo responsável.
- Run de fine-tuning: `658901685b1e42f88a0957f73a8e12e9`.
- Run de seleção/calibração: `8ab1f341b2604e19a894cfd209ca20e2`.
- Limite semântico: primeiros 256 tokens.

A ficha do checkpoint inicial informa licença MIT. Preserve as atribuições e
condições dos modelos/datasets de origem; isso não define uma licença geral
para todo o código e dados deste repositório.

## Avaliação e uso previsto

No teste de 1.659 registros, o ensemble obteve F1 macro **0,8814** e ROC AUC
**0,9606**. A versão é experimental, `production_ready=false`, para testes locais
e pesquisa. Não verifica fatos em fontes externas. As explicações de termos
representam somente o ramo léxico. Veja o [relatório](../docs/mdeberta.md).

## Outros artefatos

| Arquivo | Uso |
|---|---|
| `models/classificador.joblib` | Modo `calibrated`; exige sklearn 1.8.0 |
| `models/best_linearsvc_model.joblib` | Baseline priorizado em `legacy` |
| `models/model.pkl` | Artefato legado preservado |

O pacote `ml/` contém o motor do artefato calibrado. O hash de seu pré-processamento
é validado no carregamento; alterações precisam preservar a compatibilidade.
