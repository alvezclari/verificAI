# Modelo e métricas de avaliação do verificAI

Este documento descreve o modelo do bundle `mdeberta-ensemble-20261013T185545`,
o protocolo de avaliação e a interpretação das métricas registradas. Os valores
foram conferidos no [manifesto do modelo](../models/mdeberta-ensemble/manifest.json)
e na [procedência dos dados](../models/mdeberta-ensemble/data_provenance.json).
São resultados salvos do experimento, sem novo treinamento ou avaliação.

## 1. Modelo avaliado

O verificAI combina dois classificadores supervisionados de textos de notícias:

- **Ramo léxico:** TF-IDF e regressão logística do artefato importado
  `classificador.joblib`, incluindo sua calibração e preparação de texto.
- **Ramo semântico:** mDeBERTa-v3 ajustado para a tarefa binária, inicializado a
  partir de `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. Recebe até 256 tokens,
  incluindo tokens especiais; conteúdos maiores são truncados nesse ramo.

As probabilidades dos ramos são combinadas com pesos de 90% para o léxico e 10%
para o semântico. A combinação recebe calibração com temperatura e intercepto.
Esses pesos foram selecionados na validação; não representam a importância
causal de cada ramo. Detalhes de treinamento estão no [relatório do mDeBERTa](mdeberta.md).

O modelo estima as classes `0 = fake` e `1 = confiável`. Ele classifica padrões
textuais e não consulta fontes externas para verificar uma alegação.

## 2. Dados e protocolo

O conjunto reúne textos do Fake.Br e do `banco_fake.csv`. Após a remoção de 18
duplicatas, contém **8.295 textos: 4.713 falsos e 3.582 verdadeiros**. Os rótulos
do banco foram confirmados pelo responsável como falsos em 13/10/2026.

| Partição | Textos | Finalidade |
|---|---:|---|
| Treino | 4.976 | Ajustar os parâmetros treináveis do Transformer |
| Validação | 829 | Selecionar checkpoint e pesos da combinação |
| Calibração | 831 | Ajustar temperaturas e intercepto |
| Teste | 1.659 | Medir o desempenho após a seleção |

O particionamento usa `StratifiedGroupKFold`, seed 42 e proporções aproximadas
de 60/10/10/20. Pares do Fake.Br compartilham um grupo; URLs do banco também são
agrupadas. Duplicatas de texto normalizado e de URL são removidas antes da divisão.
Os grupos não cruzam partições do novo experimento.

O checkpoint é escolhido por F1 macro na validação. Para selecionar a combinação,
cinco pares de pesos são comparados por F1 macro na validação, com log loss como
desempate. Cada candidato tem sua calibração ajustada na reserva de calibração.
O teste entra depois dessa seleção. As métricas da reserva de calibração refletem
dados usados no ajuste e não substituem o resultado de teste.

A independência completa entre as partições históricas do classificador importado
e esse novo conjunto não foi certificada.

## 3. Decisão binária e faixas da interface

Em [metrics_for](../backend/ml/training.py), a decisão binária é:

```text
P(confiável) >= 0,5 → confiável
P(confiável) <  0,5 → fake
```

Acurácia, precisão, recall, F1 e matriz de confusão usam essa decisão. ROC AUC,
log loss e Brier usam as probabilidades, sem convertê-las em classes nesse limiar.

A interface aplica uma política com três faixas à probabilidade final calibrada:

| P(fake) | Faixa interna |
|---|---|
| Até 0,35 | Verdadeira |
| Acima de 0,35 e até 0,65 | Inconclusiva |
| Acima de 0,65 | Falsa |

“Inconclusiva” é uma política de decisão, não uma terceira classe de treinamento.
As métricas binárias publicadas incluem todos os textos, inclusive aqueles que
ficam nessa faixa. O score da interface é `100 × P(confiável)`; um score de 90
não significa que o sistema tenha 90% de acurácia global.

## 4. Métricas de classificação

Para interpretar os erros abaixo, considere **falsa como classe positiva**:

| Situação | Predita falsa | Predita confiável |
|---|---|---|
| Realmente falsa | Verdadeiro positivo (VP) | Falso negativo (FN) |
| Realmente confiável | Falso positivo (FP) | Verdadeiro negativo (VN) |

Um FP acusa uma notícia confiável de ser falsa; um FN deixa passar uma notícia
falsa. Essa convenção ajuda a discutir o custo dos erros. No código do ensemble,
a matriz é salva na ordem `[fake, confiável]`, com linhas reais e colunas preditas:
`[[VP, FN], [FP, VN]]`. Já [ml/evaluation.py](../ml/evaluation.py), usado pelo
classificador calibrado, salva a ordem `[verdadeira, falsa]`:
`[[VN, FP], [FN, VP]]`. Confira a ordem antes de comparar relatórios.

### Acurácia (`accuracy`)

```text
Acurácia = (VP + VN) / total de textos
```

Mede a proporção total de decisões corretas. Pode ocultar erros em uma classe
menos frequente; por isso deve ser lida junto das métricas por classe.

### Precisão e recall (`precision_macro`, `recall_macro`)

```text
Precisão da classe falsa = VP / (VP + FP)
Recall da classe falsa   = VP / (VP + FN)
```

A precisão responde quantas notícias marcadas como falsas eram falsas. O recall
responde quantas notícias falsas foram detectadas. O sufixo `macro` indica a
média simples dos resultados calculados separadamente para falsa e confiável;
não é a métrica apenas da classe falsa. As definições seguem o
[guia de avaliação do scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures).

### F1 macro (`f1_macro`)

```text
F1 de uma classe = 2 × precisão × recall / (precisão + recall)
F1 macro = (F1 falsa + F1 confiável) / 2
```

O F1 combina precisão e recall pela média harmônica. A média macro dá o mesmo
peso às duas classes, independentemente de suas frequências. Ela é a métrica
principal de seleção implementada no projeto. F1 macro não é a média harmônica
entre precisão macro e recall macro. A implementação usa `zero_division=0`
para termos indefinidos. Veja a [definição de F1](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.f1_score.html).

### ROC AUC (`roc_auc`)

Avalia a capacidade de ordenar as duas classes ao variar o limiar. No código,
usa `P(confiável)` e a classe `1 = confiável`. Valores maiores são melhores;
0,5 corresponde a uma ordenação sem discriminação e 1 à separação perfeita.
Um ROC AUC de 0,9606 **não significa 96,06% de notícias corretamente classificadas**
nem garante probabilidades calibradas. Veja a [definição de ROC AUC](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.roc_auc_score.html).

## 5. Qualidade das probabilidades

Nas fórmulas desta seção, `y = 1` indica notícia confiável, `y = 0` indica falsa
e `p = P(confiável)`.

### Log loss (`log_loss`)

```text
Log loss = −média[y × ln(p) + (1 − y) × ln(1 − p)]
```

Penaliza probabilidades incorretas, sobretudo erros com confiança elevada.
Quanto menor, melhor; o ideal é zero. Uma previsão constante `p = 0,5` tem
log loss de aproximadamente 0,6931. O resultado não é uma porcentagem de erro.
Veja a [definição de log loss](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.log_loss.html).

### Brier score (`brier_score`)

```text
Brier = média[(p − y)²]
```

Mede o erro quadrático das probabilidades. Na convenção binária usada pelo
projeto, varia de 0 a 1; menor é melhor. A previsão constante `p = 0,5` tem
Brier de 0,25. Um resultado de 0,0827 não equivale a 8,27% de classificações
erradas. Veja a [definição de Brier](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.brier_score_loss.html).

Log loss e Brier avaliam a qualidade probabilística, incluindo aspectos de
discriminação e calibração. A redução dessas perdas, isoladamente, não prova
que as probabilidades estejam bem calibradas. Uma curva de confiabilidade compara
probabilidade média prevista e frequência observada em grupos de previsões.
Veja o [guia de calibração](https://scikit-learn.org/stable/modules/calibration.html).

### ECE e taxas de erro do classificador calibrado

O módulo [ml/evaluation.py](../ml/evaluation.py) também calcula:

- `ece_10_bins`: divide `P(confiável)` em dez intervalos iguais e soma a diferença
  absoluta entre probabilidade média e frequência de verdadeiras, ponderada pelo
  tamanho de cada intervalo. Menor indica maior concordância nesse agrupamento.
- `fpr = FP / (FP + VN)`: proporção das notícias confiáveis acusadas de falsas.
- `fnr = FN / (FN + VP)`: proporção das notícias falsas classificadas como confiáveis.

O ECE depende dos intervalos e do tamanho da amostra. Esse módulo também verifica
ECE até 0,05, ausência de scores extremos e medianas por classe. São critérios
internos daquele fluxo. **O manifesto do ensemble documentado aqui não registra
ECE, FPR ou FNR globais nem comprova aprovação nesses critérios.**

## 6. Resultados registrados no teste

Todos os componentes abaixo foram avaliados nos mesmos **1.659 textos**. As saídas
do Transformer e do ensemble incluem a calibração descrita no experimento.
O ramo léxico corresponde à saída do artefato importado, antes da calibração
adicional da combinação. Os valores estão arredondados a quatro casas decimais.

| Métrica | Léxico importado | mDeBERTa calibrado | Ensemble calibrado | Melhor direção |
|---|---:|---:|---:|---|
| Acurácia | 0,4623 | 0,8807 | 0,8849 | Maior |
| Precisão macro | 0,7226 | 0,8818 | 0,8872 | Maior |
| Recall macro | 0,5270 | 0,8743 | 0,8779 | Maior |
| F1 macro | 0,3594 | 0,8773 | 0,8814 | Maior |
| ROC AUC | 0,9118 | 0,9531 | 0,9606 | Maior |
| Log loss | 1,5709 | 0,2781 | 0,2715 | Menor |
| Brier score | 0,4624 | 0,0866 | 0,0827 | Menor |

Fonte: `evaluation.splits.test` no [manifesto](../models/mdeberta-ensemble/manifest.json).

O ensemble acertou **88,49% das decisões binárias** nesse teste. Seu F1 macro
superou o Transformer em aproximadamente **0,0041**, ou 0,41 ponto percentual
na escala de 0 a 100. O ganho observado é pequeno; sem intervalos de confiança
ou testes de comparação, não se pode afirmar significância estatística.

O léxico combina ROC AUC relativamente alto com baixo F1 no limiar 0,5 e perdas
probabilísticas elevadas. Isso mostra que ordenação, decisão no limiar e qualidade
probabilística medem aspectos diferentes. As métricas históricas desse artefato
pertencem a outro protocolo e não devem substituir estes resultados.

### Recorte do banco com notícias falsas

O teste inclui 225 textos do `banco_fake.csv`, todos rotulados como falsos.

| Indicador do ensemble nesse recorte | Resultado |
|---|---:|
| Recall de falsas no limiar binário | 84,89% (191/225) |
| Proporção na faixa falsa, `P(fake) > 0,65` | 80,89% (182/225) |
| Proporção inconclusiva, `0,35 < P(fake) <= 0,65` | 8,00% (18/225) |

Fonte: `evaluation.banco_fake.test` no manifesto. O cálculo específico desse
recorte usa `P(fake) >= 0,5` para recall; exatamente em 0,5, seu desempate difere
da decisão global, que atribui a classe confiável nesse ponto.

Esse recorte mede detecção de falsas. Como não contém notícias verdadeiras,
não permite avaliar discriminação entre as duas classes por ROC AUC nem estimar
a taxa de falsas acusações. O F1 macro de duas classes não é apropriado para
resumir o desempenho desse recorte isoladamente.

## 7. Consulta dos resultados

Para consultar os números salvos, execute na raiz do repositório:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("models/mdeberta-ensemble/manifest.json").read_text(encoding="utf-8")
)
print("Versão:", manifest["version"])
for component, metrics in manifest["evaluation"]["splits"]["test"].items():
    print(component)
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
PY
```

Esse comando apenas lê o manifesto. Com o histórico MLflow disponível, o run
`8ab1f341b2604e19a894cfd209ca20e2` contém os relatórios em
`evaluation/{componente}_{particao}.json`, incluindo matriz de confusão e precisão,
recall, F1 e suporte por classe. Os identificadores dos componentes nesses
arquivos são `imported_lexical`, `mdeberta` e `deep_ensemble`.

Na API com esse bundle carregado, as métricas aparecem em
`model_evaluation.splits.test` na resposta de análise. São métricas do experimento
associado ao modelo, não uma avaliação da URL enviada pelo usuário.

## 8. Limitações e avaliação de próximas versões

O bundle permanece **experimental**, com `production_ready=false`. Fontes e
épocas podem se correlacionar com os rótulos; o agrupamento não exclui todas as
paráfrases. O truncamento também limita o contexto recebido pelo Transformer.
O teste de textos não mede falhas de extração de páginas na aplicação.

Para avaliar próximas versões, recomenda-se manter F1 macro como referência de
seleção e apresentar também métricas por classe, matriz de confusão, log loss,
Brier e resultados por fonte e comprimento. A política com três faixas deve
incluir taxa de inconclusivos e desempenho entre as decisões conclusivas, sempre
informando sua cobertura: `textos conclusivos / total de textos`.

Comparações devem usar o mesmo conjunto reservado e informar incerteza, por exemplo
com reamostragem dos grupos. Antes de generalizar para uso em outras condições,
é necessário avaliar fontes ou períodos externos ao desenvolvimento e esclarecer
a possível sobreposição com o treino histórico do ramo léxico. Essas avaliações
adicionais não estão registradas no manifesto atual.
