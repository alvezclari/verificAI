# Interface atual

Este diretório contém a interface HTML/CSS/JavaScript servida pelo Flask.
Execute a aplicação a partir da raiz do repositório, após instalar as dependências
e baixar os modelos conforme o [README principal](../README.md):

```bash
MODEL_BACKEND=ensemble PORT=5001 python run.py
```

Abra http://localhost:5001. Não é necessário instalar Node.js ou iniciar Vite.
`app.js` chama a API da mesma origem em `/api/analyze`; `/docs` é a documentação
interativa. Os protótipos estáticos e React anteriores estão em
[`archive/`](../archive/README.md).
