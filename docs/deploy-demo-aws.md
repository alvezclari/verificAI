# Demonstração de três horas na AWS

Proposta para minimizar o gasto de uma apresentação: Lightsail Linux/Ubuntu
**x86_64, 4 GB de RAM, com IPv4**, Docker Compose e inferência em CPU. Use o plano
de 8 GB se o ensaio de análise real ficar sem memória ou lento demais. Em ambos os
casos, faça uma análise por vez e exclua a instância ao terminar.

Os arquivos Docker foram construídos e testados localmente. Isso não cria recursos
na AWS. Repetir o build e o smoke test no servidor escolhido antes da apresentação.
Não trocar silenciosamente as versões das dependências se a instalação falhar.

## Verificação local dos arquivos de deploy

Na revisão de 15/10/2026, passaram as seguintes verificações:

- Build da aplicação para **Linux x86_64 (`linux/amd64`)**, pelo Docker Desktop
  em um Mac ARM. O Compose fixa essa arquitetura para corresponder à instância
  prevista no tutorial; a execução local usa emulação.
- Instalação do PyTorch `2.14.1+cpu` para Python 3.14 e `pip check` sem conflitos.
- Integridade dos 11 arquivos do bundle, incluindo os hashes SHA-256.
- Leitura dos pesos pelo usuário `app`, sem root. A revisão corrigiu a propriedade
  dos arquivos no Dockerfile: o peso principal local tem permissão `600` e precisa
  pertencer ao usuário da aplicação dentro da imagem.
- Smoke test com os modelos reais, extração de HTML sintético e explicações,
  usando o filesystem somente leitura e o diretório temporário do Compose.
- Inicialização pelo Gunicorn, healthcheck saudável e configuração válida do Caddy.
- Interface e API acessíveis pelo proxy; `/api/reload-model` e sua variante com
  barra final bloqueadas com HTTP 404.
- Análise completa por HTTP de uma página sintética servida na rede Docker,
  passando pelo Caddy, extração, ensemble e explicações.
- Memória observada durante a análise local: aproximadamente 794 MiB para a
  aplicação e 12 MiB para Caddy. O teste foi em um Mac com Docker Desktop e
  arquitetura x86_64 emulada; isso não é uma medição de pico nem garante o
  desempenho da instância Lightsail. O plano de 4 GB é a primeira opção de baixo
  custo a ensaiar, com o plano de 8 GB como alternativa.

Os testes HTTP locais usaram portas restritas ao próprio computador, em um projeto
Compose temporário. Não validam desempenho da AWS, acesso a portais de notícias,
concorrência de uma turma ou emissão de certificado HTTPS. Esses pontos devem ser
ensaiados no ambiente de destino conforme o escopo da apresentação.

## Custo e encerramento

Preços consultados em 15/10/2026: planos Linux de uso geral com IPv4 de 4 GB por
US$ 24/mês e 8 GB por US$ 44/mês. A cobrança é horária até o teto mensal.
Para planejamento, usando 730 horas por mensalidade:

| Tempo entre criação e exclusão | 4 GB | 8 GB |
| --- | ---: | ---: |
| 3 horas | US$ 0,10 | US$ 0,18 |
| 6 horas, incluindo preparação | US$ 0,20 | US$ 0,36 |
| 24 horas, incluindo ensaio na véspera | US$ 0,79 | US$ 1,45 |

Esses valores são aproximações do servidor, não uma cotação da fatura. Confira
a tarifa horária no console da região escolhida. Pressupõem transferência dentro
da franquia e não incluem impostos, domínio, snapshots ou outros recursos.
Os créditos só abatem serviços elegíveis e precisam estar válidos.

**Stop não encerra a cobrança do Lightsail.** Após a apresentação, excluir a
instância e conferir recursos remanescentes. Não criar snapshots ou IP estático
para esta demonstração se não forem necessários. Se criar IP estático, liberar
esse IP depois de excluir a instância; desconectado, ele pode continuar cobrando.

Fontes: [preços](https://aws.amazon.com/lightsail/pricing/),
[cobrança e exclusão](https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-frequently-asked-questions-faq-billing-and-account-management.html),
[créditos](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/useconsolidatedbilling-credits.html).

## 1. Preparar a instância

1. No console AWS, abrir **Billing and Cost Management → Credits**. Confirmar
   que o crédito está ativo e se Lightsail aparece entre os produtos elegíveis.
   Créditos promocionais variam por conta e data de expiração.
2. Abrir **Lightsail → Create instance** e escolher uma região próxima à
   apresentação. Em **Choose your instance image**, selecionar **Linux/Unix →
   OS Only → Ubuntu 24.04 LTS**. Não escolher um blueprint WordPress.
3. Em **Choose your instance plan**, selecionar Linux com **public IPv4**, plano
   de **4 GB RAM / 2 vCPU / 80 GB** (US$ 24/mês no preço consultado). O plano de
   2 GB custa menos, mas não é recomendado para esse ensemble. O plano de 8 GB
   custa US$ 44/mês e serve de alternativa caso o ensaio revele falta de memória.
4. Dar um nome fácil de reconhecer, por exemplo `verificai-demo`. Selecione uma
   chave SSH existente ou crie e baixe uma nova; guarde o `.pem` localmente e
   nunca o envie ao repositório. A chave é específica da região.
5. Clicar em **Create instance** e aguardar o estado `Running`. Registrar o nome
   e a região para a exclusão. Use o IP público que o painel mostrar; não precisa
   criar IP estático para esta apresentação.
6. Na instância, abrir **Networking → IPv4 firewall**. Liberar HTTP TCP/80 para
   os visitantes da apresentação e permitir SSH TCP/22 somente do seu IP
   público atual (formato `/32`). Não criar regra para 8000. Para o tutorial de IP
   via HTTP não é necessário liberar 443. Se IPv6 também estiver habilitado,
   confira as regras IPv6 separadamente.
7. Na aba **Connect**, usar o terminal SSH do navegador ou conectar pelo terminal
   local com a chave baixada:

   ```bash
   chmod 400 /caminho/chave-lightsail.pem
   ssh -i /caminho/chave-lightsail.pem ubuntu@IP_PUBLICO
   ```

8. Instalar Docker Engine e o plugin Compose seguindo o
   [guia oficial para Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
   No Ubuntu 24.04 recém-criado, os comandos oficiais são:

   ```bash
   sudo apt update
   sudo apt install -y ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc
   sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
   Types: deb
   URIs: https://download.docker.com/linux/ubuntu
   Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
   Components: stable
   Architectures: $(dpkg --print-architecture)
   Signed-By: /etc/apt/keyrings/docker.asc
   EOF
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   sudo systemctl status docker --no-pager
   sudo docker run hello-world
   sudo docker compose version
   ```

   Usar `sudo` nos comandos Docker deste tutorial; assim não é preciso alterar
   os grupos e permissões do usuário do servidor.

Esta configuração é para uma demonstração controlada. A aplicação ainda tem
validação de URL básica: antes de abrir análises para pessoas desconhecidas,
reforçar proteção de redes privadas/DNS/redirecionamentos e adicionar limites
de requisições. O bloqueio de `/api/reload-model` já está no Caddy.

Para uma apresentação única, o IP público basta. Não é necessário comprar domínio.
O guia usa **HTTP**, sem criptografia. Compartilhe só o endereço do site e não
envie senhas, dados pessoais ou informação confidencial. Para HTTPS é preciso um
domínio existente, liberar 80 e 443 e seguir a configuração opcional adiante.

## 2. Enviar o projeto atual

No computador local, na raiz do repositório, conferir os pesos e gerar um pacote.
O pacote inclui as alterações locais da interface e os arquivos novos de deploy,
mesmo que ainda não tenham sido enviados ao Git. Não inclui `.env` ou credenciais.

```bash
git lfs pull
python3 scripts/check_model.py --verify-hashes
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='training_args.bin' \
  -czf /tmp/verificai-demo.tar.gz \
  Dockerfile .dockerignore compose.yaml run.py \
  requirements.txt requirements-classificador.txt requirements-inference.txt \
  backend ml frontend models/mdeberta-ensemble \
  scripts/check_model.py scripts/smoke_test.py \
  deploy/Caddyfile deploy/healthcheck.py deploy/.env.example
```

O arquivo resultante contém cerca de 1,1 GiB de pesos mais código. A transferência
e o build exigem internet e espaço livre no servidor; o plano de 4 GB inclui 80 GB
de SSD.

Substituir o caminho da chave e o IP. Usar a chave SSH da região/instância escolhida;
nunca enviar essa chave para o repositório ou para o container.

```bash
scp -i /caminho/chave.pem /tmp/verificai-demo.tar.gz ubuntu@IP_PUBLICO:~/
ssh -i /caminho/chave.pem ubuntu@IP_PUBLICO
```

No servidor:

```bash
mkdir -p ~/verificai-demo
cd ~/verificai-demo
tar -xzf ~/verificai-demo.tar.gz
cp deploy/.env.example deploy/.env
```

## 3. Construir e validar antes de abrir o site

Todos os comandos a seguir são executados no servidor, em `~/verificai-demo`.

```bash
sudo docker compose --env-file deploy/.env config --quiet
sudo docker compose --env-file deploy/.env build app
sudo docker compose --env-file deploy/.env run --rm --no-deps app python scripts/smoke_test.py
sudo docker compose --env-file deploy/.env up -d --wait --wait-timeout 300
sudo docker compose --env-file deploy/.env ps
curl --fail http://localhost/api/health
```

O build instala PyTorch para CPU, preserva as versões atuais de inferência e
verifica os hashes dos pesos. Se não houver wheel compatível no índice CPU, o
build falha: resolver e validar essa dependência antes de continuar.

O smoke test usa os dois modelos reais e uma notícia sintética. Esperar
`status: ok`. Ele deve rodar **antes** de iniciar o serviço, evitando duas cópias
do modelo em memória. Para repeti-lo com a aplicação já rodando, parar primeiro
os serviços com `sudo docker compose --env-file deploy/.env stop`.

O healthcheck inspeciona o JSON de `/api/health`, incluindo ambos os modelos;
não se limita ao HTTP 200. O Caddy só inicia após a aplicação ficar saudável.
Se a aplicação falhar depois, o healthcheck sinaliza o problema, mas por si só
não reinicia um processo que continua vivo. Consultar os logs e corrigir a causa.

## 4. Compartilhar e ensaiar

Abrir de outro dispositivo autorizado no firewall:

```text
http://IP_PUBLICO
```

O Flask serve interface e API no mesmo endereço. Fechar o SSH ou desligar seu
computador não interrompe o servidor. Não existe custo por chamada a uma API de
IA externa: a inferência roda dentro do container.

Testar previamente as URLs que serão apresentadas; portais podem bloquear
requisições vindas da AWS. Analisar inicialmente uma notícia por vez e medir
tempo de resposta e memória:

```bash
sudo docker compose --env-file deploy/.env logs --tail=100 app caddy
sudo docker stats --no-stream
```

O padrão preserva as explicações (`XAI_ENABLED=true`, oito avaliações). Se forem
lentas, é possível reduzir `XAI_MAX_EVALUATIONS` no arquivo `deploy/.env` e
recriar o serviço com `up -d`. Isso reduz a cobertura da explicação e precisa de
novo ensaio. Não há benchmark de concorrência que garanta capacidade para uma turma.

## HTTPS opcional com domínio existente

Criar um registro DNS A apontando para o IP da instância e editar `deploy/.env`:

```dotenv
SITE_ADDRESS=verificai.seudominio.com
XAI_ENABLED=true
XAI_MAX_EVALUATIONS=8
```

Executar novamente:

```bash
sudo docker compose --env-file deploy/.env up -d --wait --wait-timeout 300
```

O Caddy emite e renova o certificado; `/data` e `/config` ficam em volumes Docker.
O DNS e as portas 80/443 precisam estar acessíveis para emissão do certificado.
Ver [HTTPS no Caddy](https://caddyserver.com/docs/quick-starts/https).

## 5. Encerrar a demonstração

1. Salvar localmente apenas os logs/resultados que precisar conservar.
2. No Lightsail, excluir a instância da demonstração. Conferir nome e região.
3. Se tiver criado IP estático, liberá-lo. Conferir snapshots, discos adicionais
   e outros recursos que possam permanecer cobrando.
4. Remover o registro DNS se tiver apontado um domínio para o servidor excluído.
5. Conferir o Billing depois da atualização de uso; a cobrança não aparece
   necessariamente em tempo real.

`docker compose down`, fechar o terminal e desligar a instância **não encerram
a cobrança do servidor Lightsail**. A exclusão apaga os arquivos no servidor;
o código e os pesos originais devem continuar no seu computador/repositório.

Para um ensaio na véspera e uma apresentação no dia seguinte, planejar cerca de
US$ 0,80 de consumo de créditos do servidor de 4 GB, ou US$ 1,45 no de 8 GB,
se a instância for excluída em 24 horas. É uma estimativa de compute, não um teto
de gasto; alertas de orçamento AWS notificam, mas não interrompem recursos.
