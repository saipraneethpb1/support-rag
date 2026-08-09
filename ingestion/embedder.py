"""Embedder. Three backends: sentence-transformers, fastembed, or api."""
from __future__ import annotations
import asyncio
from typing import Sequence

from config.settings import get_settings
from cache.embedding_cache import EmbeddingCache
from observability.logger import get_logger

log = get_logger(__name__)

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


class Embedder:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None
        self._cache = EmbeddingCache()
        self._backend = getattr(self._settings, "embedding_backend", "api")

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._backend == "api":
            self._model = "api"
            log.info(
                "using_api_embeddings",
                model=self._settings.effective_embedding_model_id(),
                dim=self._settings.embedding_dim,
            )
        elif self._backend == "fastembed":
            try:
                from fastembed import TextEmbedding
            except ImportError as e:
                raise RuntimeError(
                    "EMBEDDING_BACKEND=fastembed requires the 'fastembed' package. "
                    "Install it locally or switch to embedding_backend=api / sentence-transformers."
                ) from e
            log.info("loading_fastembed_model", model=self._settings.embedding_model)
            self._model = TextEmbedding(model_name=self._settings.embedding_model)
        else:
            from sentence_transformers import SentenceTransformer
            log.info("loading_st_model", model=self._settings.embedding_model)
            self._model = SentenceTransformer(self._settings.embedding_model)
        return self._model

    async def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        cached = await self._cache.get_many(list(texts))
        missing_idx = [i for i, v in enumerate(cached) if v is None]
        if missing_idx:
            to_embed = [texts[i] for i in missing_idx]
            new_vecs = await self._encode(to_embed, batch_size, task="retrieval_document")
            for i, vec in zip(missing_idx, new_vecs):
                cached[i] = vec
            await self._cache.set_many(to_embed, new_vecs)
        return cached  # type: ignore[return-value]

    async def embed_query(self, text: str) -> list[float]:
        vecs = await self._encode([text], 1, task="retrieval_query")
        return vecs[0]

    async def _encode(self, texts: list[str], batch_size: int, task: str = "retrieval_document") -> list[list[float]]:
        self._load_model()
        if self._backend == "api":
            if self._settings.google_api_key:
                return await self._encode_via_google(texts, task=task)
            return await self._encode_via_hf(texts)
        else:
            return await asyncio.to_thread(self._encode_sync, texts, batch_size)

    async def _encode_via_google(self, texts: list[str], *, task: str = "retrieval_document") -> list[list[float]]:
        from google import genai
        client = genai.Client(api_key=self._settings.google_api_key)
        dim = self._settings.embedding_dim

        async def _one(text: str) -> list[float]:
            result = await asyncio.to_thread(
                client.models.embed_content,
                model=GEMINI_EMBEDDING_MODEL,
                contents=text,
                config={"task_type": task, "output_dimensionality": dim},
            )
            return list(result.embeddings[0].values)

        return list(await asyncio.gather(*[_one(t) for t in texts]))

    async def _encode_via_hf(self, texts: list[str]) -> list[list[float]]:
        import httpx

        if not self._settings.hf_token:
            raise RuntimeError(
                "API embedding backend has no GOOGLE_API_KEY and HF_TOKEN is unset. "
                "Set GOOGLE_API_KEY (preferred) or HF_TOKEN for HuggingFace Inference."
            )
        model = self._settings.embedding_model
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self._settings.hf_token}"}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
            resp.raise_for_status()
            embeddings = resp.json()
            result = []
            for emb in embeddings:
                if isinstance(emb[0], list):
                    dim = len(emb[0])
                    pooled = [sum(emb[t][d] for t in range(len(emb))) / len(emb) for d in range(dim)]
                    result.append(_normalize(pooled))
                else:
                    result.append(_normalize(emb))
            return result

    def _encode_sync(self, texts: list[str], batch_size: int) -> list[list[float]]:
        if self._backend == "fastembed":
            embeddings = list(self._model.embed(texts, batch_size=batch_size))
            return [e.tolist() for e in embeddings]
        else:
            arr = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return arr.tolist()


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]
