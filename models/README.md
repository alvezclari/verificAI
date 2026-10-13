# Artefatos dos classificadores

`best_linearsvc_model.joblib` e `model.pkl` são os artefatos legados. A API
prioriza o primeiro quando `MODEL_BACKEND=legacy`.

`MODEL_BACKEND=calibrated` carrega `classificador.joblib` usando o pacote
`ml/`, com calibração, ajuste para texto curto e faixas originais. Execute com
`requirements-classificador.txt`. Veja o [guia da API](../docs/api-local.md).

O treinamento novo gera `models/ensemble/lexical.joblib`, `lexical_metadata.json`,
`transformer/`, `data_provenance.json` e `manifest.json`. O Transformer contém
pesos, configuração, tokenizador e `training_metadata.json` exportados após
fine-tuning. O manifesto é criado após avaliar os dois componentes.

O bundle é ignorado pelo Git e registrado no MLflow. Não há pipeline DVC
preexistente; caso ele seja adotado, versione o diretório inteiro para manter
os artefatos consistentes. O novo CLI não sobrescreve os arquivos legados.

Consulte [treinamento, ativação e convenções de classes](../docs/ensemble.md).


O bundle ativo local é `models/mdeberta-ensemble/`: cópia do classificador
importado, pesos/tokenizador do mDeBERTa supervisionado, procedência e manifesto
v2 com pesos e calibração do ensemble. Quando esse manifesto existe, ele é o
padrão na inicialização sem `MODEL_BACKEND` explícito. Veja o
[guia do mDeBERTa](../docs/mdeberta.md).

Para restaurar esses pesos em outro ambiente, use o [guia de transferência](../docs/continuacao-em-outro-ambiente.md).
O script `scripts/transfer_artifacts.py` inclui o bundle completo, dados e
MLflow, com checksums. Os arquivos em `exports/` são transferidos separadamente
do repositório; não são enviados pelo Git.
