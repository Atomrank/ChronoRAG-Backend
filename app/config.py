from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-10-21"
    azure_chat_deployment: str = "gpt-4o"
    azure_embed_deployment: str = "text-embedding-3-small"
    embed_dim: int = 1536

    # Stores
    pg_dsn: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Pipeline
    concurrency: int = 5
    window_size: int = 10
    window_overlap: int = 2
    naive_chunk_chars: int = 1200
    naive_chunk_overlap: int = 200
    naive_top_k: int = 8
    pass2_batch_size: int = 2

    # LLM input budget. Inputs above this raise InputTooLongError instead of
    # being silently cut (v1 cut every input at 14,000 chars).
    llm_max_input_chars: int = 100_000

    # v2 ingestion / windows
    v2_window_chars: int = 12_000
    v2_window_overlap_paras: int = 2
    v2_extract_samples: int = 1          # 3 for reported runs (relation self-consistency)
    local_llm_base_url: str | None = None   # OpenAI-compatible server (vLLM) for filtered windows
    local_llm_model: str = ""

    # Evaluation
    eval_k_max: int = 50
    # Gold proposal must use a DIFFERENT, long-context model than the pipeline
    azure_gold_deployment: str = "gpt-4.1"
    gold_max_input_chars: int = 3_000_000
    eval_dir: str = "./data/eval"

    # Paths
    upload_dir: str = "./data/uploads"
    cache_dir: str = "./data/cache"

    @property
    def upload_path(self) -> Path:
        p = (ROOT / self.upload_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def eval_path(self) -> Path:
        p = (ROOT / self.eval_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def public_dict(self) -> dict:
        """All settings with secrets removed — stored with every eval run."""
        d = self.model_dump()
        for k in list(d):
            if any(x in k for x in ("key", "password", "dsn")):
                d[k] = "***"
        return d

    @property
    def cache_path(self) -> Path:
        p = (ROOT / self.cache_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()

# Fallback only — used if Pass 0 has not produced a per-document taxonomy yet.
# Prefer document.taxonomy from the DB for any real build.
DEFAULT_TAXONOMY: list[str] = [
    "Beginning / Setup",
    "Rising Action",
    "Midpoint / Complication",
    "Climax",
    "Resolution / Aftermath",
]

# Back-compat alias for older imports / health endpoints.
TAXONOMY = DEFAULT_TAXONOMY


def stage_names(taxonomy: list) -> list[str]:
    """Normalize taxonomy records (strings or {name,...} dicts) to stage names."""
    names: list[str] = []
    for item in taxonomy or []:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(getattr(item, "name", "") or "").strip()
        if name:
            names.append(name)
    return names or list(DEFAULT_TAXONOMY)


def stage_index(anchor: str, taxonomy: list | None = None) -> int:
    """Map a free-text anchor onto the given taxonomy, tolerating minor LLM drift."""
    names = stage_names(taxonomy if taxonomy is not None else DEFAULT_TAXONOMY)
    order = {name: i for i, name in enumerate(names)}
    if anchor in order:
        return order[anchor]
    low = (anchor or "").lower().strip()
    if not low:
        return min(1, len(names) - 1)
    for name, idx in order.items():
        nl = name.lower()
        if nl in low or low in nl:
            return idx
    # Token overlap fallback (no book-specific keyword table).
    tokens = {t for t in low.replace("/", " ").replace("-", " ").split() if len(t) > 2}
    best_idx, best_score = 0, 0
    for name, idx in order.items():
        ntokens = {t for t in name.lower().replace("/", " ").replace("-", " ").split() if len(t) > 2}
        score = len(tokens & ntokens)
        if score > best_score:
            best_idx, best_score = idx, score
    if best_score > 0:
        return best_idx
    return min(1, len(names) - 1)
