<<<<<<< HEAD
# VerificAI

> **Antes de compartilhar, observe os sinais.**

A **VerificAI** é um protótipo de aplicação voltado à **avaliação da confiabilidade de informações e notícias**. A proposta é auxiliar o usuário na análise crítica de uma informação antes de seu compartilhamento, organizando diferentes sinais de evidência em um **indicador geral de confiabilidade**.

O sistema recebe o link de uma notícia e apresenta uma análise estruturada com base em cinco critérios, permitindo que o usuário compreenda **por que uma informação recebeu determinada avaliação**, em vez de depender apenas de uma classificação automática.

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

# VerificAI - Detector de Fake News e Análise de Confiabilidade de Notícias

Aplicação completa (Fullstack) de análise crítica de notícias e detecção de Fake News desenvolvida com **Flask**, **Web Scraping (Trafilatura/BS4)**, **Machine Learning (LinearSVC treinado no FakeRecogna)** e **Frontend VerificAI (HTML5/CSS3/JS)**.

O sistema recebe a URL de uma notícia colocada pelo usuário no frontend, realiza a extração do texto jornalístico, submete o conteúdo ao classificador `LinearSVC` e decompõe a confiabilidade em uma escala de 0 a 100%, detalhando 5 critérios analíticos: **Evidências**, **Qualidade das fontes**, **Corroboração**, **Contexto** e **Atualidade**.

---

## 🚀 Como Executar a Aplicação Completa

### 1. Ativar o Ambiente Virtual
```bash
# macOS / Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate
```

### 2. Instalar Dependências (se ainda não instaladas)
```bash
pip install -r requirements.txt
```

### 3. Iniciar o Servidor Integrado
```bash
python run.py
```

### 4. Acessar no Navegador
Abra seu navegador em:
👉 **`http://localhost:5000`**

A interface do **VerificAI** carregará automaticamente na tela, conectada diretamente à API de Web Scraping e ao modelo LinearSVC!

> **Nota de flexibilidade**: Você também pode abrir o arquivo `verificAI-front_inicial/index.html` diretamente com dois cliques no navegador caso prefira; ele se comunicará perfeitamente com a API através do suporte a CORS habilitado.

---

## 🧠 Arquitetura do Modelo LinearSVC

- **Modelo**: `models/best_linearsvc_model.joblib` (Pipeline Scikit-Learn com `TfidfVectorizer` de 50.000 termos/bigramas + `LinearSVC`).
- **Treinamento**: Corpus *FakeRecogna / Fake.br*.
- **Calibração de Saída**: Distância do hiperplano (`decision_function`) convertida para porcentagem de confiança de 0 a 100% via função sigmoide:
  $$P(\text{Confiável}) = \frac{100}{1 + e^{\text{margem}}}$$
- **Regra de Negócio**:
  - $\ge 50.0\% \rightarrow$ **Confiável** (`"confiavel"`)
  - $< 50.0\% \rightarrow$ **Não Confiável / Suspeita de Fake News** (`"desconfiavel"`)

---

## 📡 Endpoints da API

- **`GET /`**: Interface Web do VerificAI (Frontend completo).
- **`GET /api/health`**: Verifica status da API, critérios e se o modelo LinearSVC está carregado.
- **`POST /api/analyze`**: Recebe `{"url": "https://..."}` e retorna a análise com score de 0 a 100, veredito e os 5 critérios.
- **`POST /api/reload-model`**: Recarrega modelos da pasta `models/` sem reiniciar o servidor.

---

## 🧪 Testes Automatizados

Para rodar todos os testes unitários e de integração:
```bash
pytest
```
Todos os 8 testes cobrem rotas da API, serviço estático do frontend, web scraping, classificação de notícias confiáveis vs fake news e calibração do LinearSVC.
>>>>>>> ab2981b (feat: make VerificAI logo clickable to return to home screen)
