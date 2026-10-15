# Interface atual

Este diretório contém a interface HTML/CSS/JavaScript servida pelo Flask.

**Como interpretar** apresenta o resumo e os trechos de
`ml_details.explicacao.local`, com efeitos favoráveis às duas classes e cobertura
da análise. Respostas sem esse campo usam a interpretação anterior. Os trechos
são inseridos como texto, sem executar HTML da notícia. Veja o
[guia de explicabilidade](../docs/explicabilidade.md).
Execute a aplicação a partir da raiz do repositório, após instalar as dependências
e baixar os modelos conforme o [README principal](../README.md):

```bash
MODEL_BACKEND=ensemble PORT=5001 python run.py
```

Abra http://localhost:5001. Não é necessário instalar Node.js ou iniciar Vite.
`app.js` chama a API da mesma origem em `/api/analyze`; `/docs` é a documentação
interativa. Os protótipos estáticos e React anteriores estão em
[`archive/`](../archive/README.md).

Após a análise de uma URL, o resultado principal mostra a estimativa de
confiabilidade e a interpretação. O painel **Sinais auxiliares do texto** fica
logo acima de **Informações técnicas**, ocupando a mesma largura e com acesso
independente para o usuário.
O painel **Informações técnicas** fica separado e recolhido a cada nova análise
e reúne probabilidades, componentes do modelo, limiares, versão, métricas de
avaliação e limitações das explicações. Os cartões de termos não são exibidos. A resposta da API
continua disponível integralmente para integrações.
