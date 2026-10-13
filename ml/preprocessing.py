"""Preparação única para título e corpo, sem remoção de números ou negações."""

import re
import unicodedata
from html.parser import HTMLParser


class _TextoHTML(HTMLParser):
    """Remove marcação residual e conteúdo de scripts/estilos."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.partes = []
        self.ignorar = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignorar += 1
        self.partes.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.ignorar:
            self.ignorar -= 1
        self.partes.append(" ")

    def handle_data(self, data):
        if not self.ignorar:
            self.partes.append(data)


def preparar_texto(titulo: str | None, corpo: str | None) -> str:
    """Combina título e corpo; preserva moeda, pontuação, datas e conteúdo lexical."""
    partes = []
    for valor in (titulo, corpo):
        if valor is not None and not isinstance(valor, str):
            raise TypeError("Título e corpo devem ser texto ou None.")
        parser = _TextoHTML()
        parser.feed(valor or "")
        parser.close()
        partes.append("".join(parser.partes))
    texto = unicodedata.normalize("NFC", " ".join(partes)).lower()
    return re.sub(r"\s+", " ", texto).strip()
