import logging
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import trafilatura
from app.config import Config

logger = logging.getLogger(__name__)

class ScraperService:
    def __init__(self, timeout: int = None, user_agent: str = None):
        self.timeout = timeout or Config.SCRAPER_TIMEOUT
        self.headers = {
            "User-Agent": user_agent or Config.SCRAPER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        }

    def scrape(self, url: str) -> dict:
        """
        Realiza o scraping da URL informada e retorna metadados e o conteúdo do artigo.
        """
        domain = urlparse(url).netloc.lower()

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            html_content = response.text
        except requests.exceptions.Timeout:
            logger.error(f"Timeout ao acessar URL: {url}")
            raise ValueError("O servidor da notícia demorou muito para responder (timeout).")
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao baixar página {url}: {e}")
            raise ValueError(f"Não foi possível acessar a URL informada: {str(e)}")

        # 1. Extração do conteúdo principal com Trafilatura
        trafilatura_result = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            output_format="json",
            with_metadata=True
        )

        title = None
        text = None
        author = None
        publish_date = None

        if trafilatura_result:
            import json
            try:
                data = json.loads(trafilatura_result)
                title = data.get("title")
                text = data.get("text")
                author = data.get("author")
                publish_date = data.get("date")
            except Exception as e:
                logger.warning(f"Erro ao decodificar resultado do trafilatura: {e}")

        # 2. Complemento/Fallback com BeautifulSoup para metadados detalhados
        soup = BeautifulSoup(html_content, "html.parser")

        # Título
        if not title:
            og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)
            else:
                title = "Título não identificado"

        # Autor
        if not author:
            meta_author = (
                soup.find("meta", attrs={"name": "author"}) or
                soup.find("meta", property="article:author") or
                soup.find(attrs={"rel": "author"})
            )
            if meta_author and meta_author.get("content"):
                author = meta_author["content"].strip()
            elif meta_author and meta_author.get_text():
                author = meta_author.get_text(strip=True)
            else:
                # Procura por elementos com classe ou item de autor
                author_elem = soup.select_one(".author, .byline, [itemprop='author']")
                if author_elem:
                    author = author_elem.get_text(strip=True)

        # Data de publicação
        if not publish_date:
            meta_time = (
                soup.find("meta", property="article:published_time") or
                soup.find("meta", attrs={"name": "date"}) or
                soup.find("meta", attrs={"name": "pubdate"}) or
                soup.find("time")
            )
            if meta_time:
                publish_date = meta_time.get("content") or meta_time.get("datetime") or meta_time.get_text(strip=True)

        # Texto do artigo (fallback se trafilatura não extraiu texto suficiente)
        if not text or len(text.strip()) < 50:
            # Procura os parágrafos dentro de <article> ou <main> ou <body>
            article_container = soup.find("article") or soup.find("main") or soup.body
            if article_container:
                # Remove scripts, styles, forms, navs
                for tag in article_container(["script", "style", "nav", "footer", "aside", "form"]):
                    tag.decompose()
                paragraphs = [p.get_text(strip=True) for p in article_container.find_all("p") if len(p.get_text(strip=True)) > 25]
                text = "\n\n".join(paragraphs)

        if not text or len(text.strip()) < 40:
            raise ValueError("Não foi possível extrair o texto da notícia. A página pode conter bloqueio, paywall ou conteúdo restrito.")

        # Contagem de links internos e externos citados
        article_links = [a.get("href") for a in soup.find_all("a", href=True) if a["href"].startswith("http")]

        word_count = len(text.split())

        return {
            "url": url,
            "domain": domain,
            "title": title.strip() if title else "Sem título",
            "text": text.strip(),
            "author": author.strip() if author else "Não informado / Redação",
            "publish_date": publish_date.strip() if publish_date else None,
            "word_count": word_count,
            "external_links_count": len(article_links),
        }

