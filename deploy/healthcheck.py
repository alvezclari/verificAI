"""Readiness do container: HTTP 200 sozinho não comprova que o ensemble carregou."""

import json
import sys
from urllib.request import urlopen


def main():
    try:
        with urlopen("http://127.0.0.1:8000/api/health", timeout=5) as response:
            status = json.load(response)
        model = status.get("model") or {}
        transformer = model.get("transformer") or {}
        ready = (
            status.get("status") == "online"
            and status.get("ensemble_loaded") is True
            and model.get("backend") == "ensemble"
            and model.get("loaded") is True
            and transformer.get("loaded") is True
        )
        return 0 if ready else 1
    except (OSError, ValueError, TypeError, AttributeError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
