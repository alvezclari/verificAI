# Desenvolvimento e envio ao GitHub

## Ambiente e verificações

Use Python 3.14 e Git LFS. O ambiente validado é Python 3.14.6 no macOS arm64.

```bash
git lfs install --local
git lfs pull
python3.14 -m venv .venv-classificador
source .venv-classificador/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check
python scripts/check_model.py --verify-hashes
python scripts/smoke_test.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

No PowerShell, ative `.venv-classificador\Scripts\Activate.ps1` e defina
`$env:HF_HUB_OFFLINE="1"` e `$env:TRANSFORMERS_OFFLINE="1"` antes dos testes.
Node.js é opcional para `node --check frontend/app.js`; não é necessário à API.

## Estrutura

- A interface atual está em `frontend/` e a API em `backend/`.
- Os protótipos anteriores foram preservados em `archive/`.
- Mantenha `ml/` compatível com o classificador: o hash de `preprocessing.py`
  participa da validação do artefato.
- Preserve as regras de bytes e quebras de linha em `.gitattributes`; os hashes
  dos artefatos devem permanecer iguais também em clones Windows.
- Registre métricas junto ao corpus, às partições e à versão que as produziram.
- `.env`, ambientes virtuais, caches, MLflow e exportações são locais.

## Preparar o primeiro envio com pesos NLP

Os arquivos grandes já estão declarados em `.gitattributes`. Não é necessário
reescrever o histórico nem migrar os artefatos antigos. O bundle mDeBERTa era
ignorado e passa a entrar no próximo commit.

```bash
git lfs install --local
git add .gitattributes .gitignore
git add models/mdeberta-ensemble
git add README.md CONTRIBUTING.md requirements-inference.txt .env.example
git add scripts/check_model.py scripts/smoke_test.py tests/test_model_distribution.py models/README.md
git add -A archive frontend docs
git add -u
git lfs ls-files
git diff --cached --stat
git diff --cached --check
```

`git lfs ls-files` deve listar `model.safetensors`, `spm.model` e `tokenizer.json`.
Confirme que o peso foi adicionado como ponteiro pequeno:

```bash
git show :models/mdeberta-ensemble/transformer/model.safetensors
```

A saída deve conter `version https://git-lfs.github.com/spec/v1`, `oid sha256:`
e `size 1115268200`, em vez do binário. Os arquivos reais permanecem no diretório
de trabalho e no armazenamento local do LFS.

Depois de revisar as mudanças:

```bash
git commit -m "chore: organizar repositorio e distribuir modelo NLP com Git LFS"
git push -u origin modularization
# Se main continuar sendo ancestral desta versão, atualize-a sem force push:
git push origin HEAD:main
```

O hook `pre-push` instalado pelo LFS envia os objetos ao remoto. Não ignore esse
hook. O armazenamento e os downloads seguem as
[cotas do Git LFS](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).
Um envio de código bem-sucedido não substitui a validação dos pesos no remoto.

## Validar o que outras pessoas receberão

Após o envio, faça um clone em outro diretório:

```bash
git clone https://github.com/alvezclari/verificAI.git ../verificai-teste
cd ../verificai-teste
git lfs pull
python3.14 scripts/check_model.py --verify-hashes
# Após instalar requirements-inference.txt no ambiente novo:
python scripts/smoke_test.py
```

Continue com a instalação e execução do README. A checagem deve passar sem
depender de `.hf-cache/`, `mlruns/`, `data/` ou arquivos da máquina original.
No health, confira `model.backend=ensemble`, `ensemble_loaded=true` e
`model.transformer.loaded=true`.

## Atualizar o bundle

Treine em outro diretório, conforme o [guia](docs/mdeberta.md), e avalie antes de
substituir o bundle publicado. Atualize pesos, configurações, manifesto,
metadados, `checksums.json` e documentação no mesmo commit. O inventário usa
SHA-256 e tamanho em bytes de cada arquivo, exceto ele próprio. Preserve a
versão anterior no histórico. Novos arquivos de pesos exigem regras LFS em
`.gitattributes` e exceções correspondentes em `.gitignore`.
