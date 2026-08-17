# What this is

This app answers questions using **only files you give it**. It is not
tied to a company, product, or support desk. Put your notes, docs, or
exports in the `data/` folder, ingest them, then ask.

## What happens when you ask

1. Your question is searched against the ingested files (keyword +
   meaning).
2. The best matching passages are sent to a language model.
3. The reply cites those passages. If nothing relevant was ingested,
   it should say so instead of guessing.

## What it is not

It does not browse the web. It does not know your files until you
ingest them. Changing a file on disk does nothing until you run ingest
again (`POST /ingest/run` or `python -m scripts.bootstrap_index`).
