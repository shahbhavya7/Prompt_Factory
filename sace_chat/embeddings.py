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


def get_embedder():
    mode = os.environ.get("EMBEDDING_MODE", "mock").lower()
    if mode == "openai":
        return OpenAIEmbedder()
    return MockEmbedder()
