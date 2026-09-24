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
    # Legacy page-window knobs (naive / older callers). Extract paths use
    # v2_window_* below for BOTH kaalkram_v1 and kaalkram_v2.
    window_size: int = 10
    window_overlap: int = 2
    naive_chunk_chars: int = 1200
    naive_chunk_overlap: int = 200
    naive_top_k: int = 8
    pass2_batch_size: int = 2

    # LLM input budget. Inputs above this raise InputTooLongError instead of
    # being silently cut (v1 cut every input at 14,000 chars).
    llm_max_input_chars: int = 100_000

    # Shared extract windowing (kaalkram_v1 pass1 + kaalkram_v2).
    # Defaults sized for dense ~1-event/paragraph text (e.g. synth).
    v2_window_chars: int = 1_500        # char budget per window (plus overlap context)
    v2_window_overlap_paras: int = 1    # preceding paras as read-only context
    v2_window_max_paras: int = 12       # also flush by paragraph count
    v2_extract_samples: int = 1         # 3 for reported runs (relation self-consistency)
    # Tokens reserved per event mention in the structured extract JSON (~quote,
    # description, participants). Output budget = 2x * max_paras * this.
    extract_tokens_per_event: int = 400
    # Floor / override; effective budget is max(this, 2*max_paras*tokens_per_event).
    v2_extract_max_tokens: int = 9_600
    # exhaustive = every narrated event (synth / recall); salient = plot-milestone skim.
    extraction_mode: str = "exhaustive"
    # llm = WindowExtraction via chat_structured; oracle = regex parse of synth templates.
    extractor: str = "llm"
    # If owned paras >> extracted mentions, split the window and re-extract.
    v2_extract_min_mention_ratio: float = 0.5
    v2_extract_split_min_paras: int = 4
    v2_entity_cosine: float = 0.72       # embed nearest-neighbour threshold for entity candidates
    v2_entity_nn_top: int = 5            # embedding NN neighbours considered per form
    v2_entity_max_group: int = 40        # split candidate groups larger than this before LLM
    v2_entity_context_chars: int = 150   # ±chars of context collected per surface form
    v2_entity_max_contexts: int = 3      # max short contexts kept per form occurrence
    v2_coref_cosine: float = 0.78        # description-embedding threshold for coref candidates
    v2_coref_top: int = 10               # max earlier-mention coref candidates per mention
    v2_coref_passage_chars: int = 300    # ±chars of passage sent to coref LLM
    v2_ground_embed_top: int = 20        # embedding candidates when grounding question events
    v2_before_after_sim_top: int = 10    # keep related events by similarity for before_after_x
    v2_passage_pad_chars: int = 200      # ±chars around mention spans for verbalise / factual
    v2_factual_top_k: int = 8            # hybrid retrieval depth for factual questions
    v2_factual_embed_weight: float = 0.6 # blend weight for embedding vs lexical in factual hybrid
    local_llm_base_url: str | None = None   # OpenAI-compatible server (vLLM) for filtered windows
    local_llm_model: str = ""

    # Evaluation
    eval_k_max: int = 50
    # Stable hash of question id: fraction assigned to the calibration / tuning split.
    eval_dev_fraction: float = 0.3
    # Mention span vs gold evidence: intersection / min(lengths) must be >= this to map.
    eval_calibrate_overlap_frac: float = 0.5
    # Gold proposal must use a DIFFERENT, long-context model than the pipeline
    azure_gold_deployment: str = "gpt-4.1"
    gold_max_input_chars: int = 3_000_000
    # Automatic gold verification: comma-separated judge deployments. "local:<model>" uses
    # LOCAL_LLM_BASE_URL. Prefer judges from DIFFERENT model families.
    gold_judges: str = "gpt-4o"
    # Optional: a judge that can read the WHOLE document; required to keep
    # cannot_determine gold labels (local passages cannot prove the text never links two events).
    gold_fulltext_judge: str = ""
    eval_dir: str = "./data/eval"

    # Synthetic benchmark defaults (CLI can override)
    synth_default_events: int = 150
    synth_default_flashback_rate: float = 0.2
    synth_default_nesting: int = 2
    synth_default_prophecy_rate: float = 0.05
    synth_default_hypothetical_rate: float = 0.05
    synth_default_separate_tale_rate: float = 0.05
    synth_default_parallel_rate: float = 0.1

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

    def extraction_max_tokens(self) -> int:
        """Output token budget for one extract window: 2× headroom at 1 event/para."""
        needed = 2 * self.v2_window_max_paras * self.extract_tokens_per_event
        return max(int(self.v2_extract_max_tokens), int(needed))


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
