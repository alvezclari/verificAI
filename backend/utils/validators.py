from urllib.parse import urlparse
import ipaddress
import socket

def validate_url(url: str) -> tuple[bool, str | None]:
    """
    Valida se a URL fornecida é válida, tem esquema http/https e possui host válido.
    Retorna (is_valid, error_message).
    """
    if not url or not isinstance(url, str):
        return False, "A URL não pode estar vazia."

    url = url.strip()
    if len(url) > 2048:
        return False, "URL muito longa (máximo de 2048 caracteres)."

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Formato de URL inválido."

    if parsed.scheme not in ("http", "https"):
        return False, "A URL deve começar com http:// ou https://"

    if not parsed.netloc:
        return False, "A URL fornecida não contém um domínio válido."

    # Bloqueia acessos a localhost / loopback por segurança (SSRF protection básica)
    hostname = parsed.hostname
    if not hostname:
        return False, "Domínio ausente na URL."

    if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return False, "Não é permitido analisar URLs locais (localhost)."

    return True, None

