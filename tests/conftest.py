import sys
from pathlib import Path

# Adiciona o diretório raiz ao sys.path para importação de 'app'
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

