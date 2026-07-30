import hashlib
import os
import struct


class MockEmbedder:
    """Deterministic offline embedder: hashes words of the text into a
    fixed-size vector so texts sharing vocabulary drift toward similar vectors
    without needing a real model or network access."""

    dim = 384

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for w in text.lower().split():
            h = hashlib.sha256(w.encode("utf-8")).digest()
            idx = struct.unpack("I", h[:4])[0] % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class OpenAIEmbedder:
    dim = 1536

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ["SACE_LLM_KEY"]
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return resp.data[0].embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """One request for a batch. Used for the intent exemplars (dozens of
        short strings warmed up at boot) and for the two query vectors a turn
        needs — sequentially those are dozens of round-trips of dead air."""
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        # The API does not promise ordering, but it does return an index.
        return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]


def embed_many(embedder, texts: list[str]) -> list[list[float]]:
    """Batch through the embedder if it supports it, else one at a time."""
    fn = getattr(embedder, "embed_many", None)
    if fn is not None:
        return fn(texts)
    return [embedder.embed(t) for t in texts]


class LocalEmbedder:
    """A local sentence-transformer, for the hot path only.

    Its dimension will NOT match the pgvector column (all-MiniLM-L6-v2 is 384,
    the stored KB vectors are OpenAI 1536), so it can never be used for a
    pgvector query. See get_hotpath_embedder for exactly where it is allowed.
    """

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.environ.get(
            "EMBED_HOTPATH_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = SentenceTransformer(self.model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]


def get_embedder():
    """The KB embedder. Everything that touches pgvector uses THIS one, because
    its dimension has to match the `embedding` column."""
    mode = os.environ.get("EMBEDDING_MODE", "mock").lower()
    if mode == "openai":
        return OpenAIEmbedder()
    return MockEmbedder()


def get_hotpath_embedder(kb_embedder=None):
    """The embedder used ONLY for intent-exemplar matching on the live turn.

    Why a second embedder exists: intent detection compares the caller's message
    against kb.INTENT_EXEMPLARS. That is a pure in-process cosine against vectors
    this module produced itself — it never reaches Postgres — so it does not need
    to share the KB's dimension. With EMBED_HOTPATH=local it runs on-device and
    removes one network round-trip from every turn, which is dead air in a voice
    call.

    WHICH EMBEDDER IS USED WHERE — this split is the whole point:

      * pgvector queries (retrieve._fetch_by_intent / _fetch_general), rule cues
        written by load_kb.py and consolidator.py, and reply grounding in
        engine.score_reply  ->  ALWAYS get_embedder() (OpenAI 1536).
        A local 384-dim vector here is a hard Postgres error, and
        db.check_embedding refuses to store one.

      * IntentRouter exemplar matching  ->  get_hotpath_embedder(), which may be
        local. Both sides of that comparison (the exemplars and the message) come
        from the same model, so the dimension is self-consistent and lower
        quality only costs intent-routing accuracy, never a query failure.

    Falls back to the KB embedder if the mode is not 'local' or if
    sentence-transformers is not installed, so a missing optional dependency
    degrades to the working path instead of breaking the call.
    """
    mode = os.environ.get("EMBED_HOTPATH", "openai").lower()
    if mode != "local":
        return kb_embedder or get_embedder()
    try:
        local = LocalEmbedder()
        print(f"[embed] hot path: local {local.model_name} ({local.dim}d) — "
              f"intent exemplars only, never pgvector")
        return local
    except Exception as exc:
        print(f"[embed] EMBED_HOTPATH=local requested but unavailable "
              f"({type(exc).__name__}: {exc}); falling back to the KB embedder")
        return kb_embedder or get_embedder()
