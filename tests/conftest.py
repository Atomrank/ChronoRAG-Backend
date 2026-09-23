import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# dummy settings so app.config imports without a real .env
for k, v in {"AZURE_OPENAI_ENDPOINT": "https://x", "AZURE_OPENAI_API_KEY": "x",
             "PG_DSN": "postgresql://x", "NEO4J_URI": "bolt://x", "NEO4J_USER": "x",
             "NEO4J_PASSWORD": "x"}.items():
    os.environ.setdefault(k, v)
