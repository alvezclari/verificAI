"""
Script de Exemplo: Treinamento e Exportação de Modelo de Fake News para a pasta models/

Este script demonstra como treinar um classificador do Scikit-Learn com TF-IDF
com splits e métricas no MLflow, salvando em models/ensemble/lexical.joblib.
"""

import sys

from backend.ml.training import parse_args, run_training

# Exemplo de mini-dataset em Português
# Rótulo 1: Notícia Confiável
# Rótulo 0: Fake News / Desconfiável
training_texts = [
    # Exemplos confiáveis
    "Pesquisa da Fiocruz aponta aumento na vacinação com dados do Ministério da Saúde",
    "Estudo da Universidade de São Paulo demonstra eficácia de 90% em novo medicamento para controle glicêmico",
    "IBGE divulga taxa de inflação referente ao último trimestre com dados oficiais e relatórios",
    "Ministério da Educação publica edital do Enem com cronograma detalhado e diretrizes de inscrição",
    "Cientistas da NASA confirmam descoberta de novo exoplaneta com base em dados de telescópio espacial",
    "Segundo relatório da OMS, medidas preventivas reduziram a incidência da doença em 30 países",
    
    # Exemplos de fake news / desconfiáveis
    "URGENTE! A grande mídia esconde a cura secreta do câncer descoberta por médico anônimo",
    "BOMBA! Compartilhe antes que apaguem! O segredo que os governos mundiais não querem que você saiba",
    "Inacreditável milagre! Chá caseiro cura todas as doenças em menos de 24 horas sem remédios",
    "Veja o vídeo chocante antes que saia do ar! Conspiração absurda é desmascarada agora",
    "Alerta urgente! Eles vão bloquear todas as contas bancárias amanhã, espalhe para todos os grupos",
    "O que ninguém te conta sobre o plano secreto que vai mudar tudo, veja a verdade oculta"
]

training_labels = [
    1, 1, 1, 1, 1, 1,  # Confiáveis
    0, 0, 0, 0, 0, 0   # Fake News
]

def train_and_export(argv=None):
    """Demonstra splits, métricas e MLflow; não mede qualidade em notícias reais."""
    args = parse_args([
        "--data", "embedded-example", "--real-label", "1", "--stage", "lexical",
        *(sys.argv[1:] if argv is None else argv),
    ])
    if args.stage != "lexical":
        raise ValueError("Use train_ensemble.py e um corpus real para treinar o Transformer.")
    return run_training(args, training_texts, training_labels, {
        "source": "embedded-example", "real_label": "1", "purpose": "smoke-test-only",
    })


if __name__ == "__main__":
    train_and_export()
