# How answers work

Retrieval is **hybrid**: vector search (meaning) plus BM25 (exact words
like error codes and filenames), fused with Reciprocal Rank Fusion.

## Citations

Every factual sentence should include markers like [1] that map to
ingested files. The UI shows those as links under the answer. Invented
markers (numbers that were not in the retrieved context) are stripped
and counted as a hallucination signal.

## Cache

Near-duplicate questions can return a cached answer. After a successful
ingest that changed the corpus, that cache is invalidated.

## Honesty

The model is instructed to refuse when the ingested files do not
contain the answer. If it still guesses, treat citations as the
source of truth and re-ingest better docs.
