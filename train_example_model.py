"""
Script de Exemplo: Treinamento e Exportação de Modelo de Fake News para a pasta models/

Este script demonstra como treinar um classificador do Scikit-Learn com TF-IDF
e salvá-lo como 'models/model.pkl', compatível diretamente com o backend.
"""

from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

def train_and_export():
    print("[*] Treinando pipeline com TfidfVectorizer + LogisticRegression...")
    
    # Monta pipeline completo que aceita texto cru
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000
        )),
        ("clf", LogisticRegression(random_state=42))
    ])
    
    pipeline.fit(training_texts, training_labels)
    
    # Salva o pipeline diretamente em models/model.pkl
    output_path = MODELS_DIR / "model.pkl"
    joblib.dump(pipeline, output_path)
    
    print(f"[+] Modelo de exemplo salvo com sucesso em: {output_path}")
    print("[+] Agora o backend carregará este modelo automaticamente!")

if __name__ == "__main__":
    train_and_export()

